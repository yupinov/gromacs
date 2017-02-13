/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 2012,2013,2014,2015,2016,2017, by the GROMACS development team, led by
 * Mark Abraham, David van der Spoel, Berk Hess, and Erik Lindahl,
 * and including many others, as listed in the AUTHORS file in the
 * top-level source directory and at http://www.gromacs.org.
 *
 * GROMACS is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License
 * as published by the Free Software Foundation; either version 2.1
 * of the License, or (at your option) any later version.
 *
 * GROMACS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with GROMACS; if not, see
 * http://www.gnu.org/licenses, or write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA.
 *
 * If you want to redistribute modifications to GROMACS, please
 * consider that scientific software is very special. Version
 * control is crucial - bugs must be traceable. We will be happy to
 * consider code for inclusion in the official distribution, but
 * derived work must not be called official GROMACS. Details are found
 * in the README & COPYING files - if they are missing, get the
 * official version at http://www.gromacs.org.
 *
 * To help us fund GROMACS development, we humbly ask that you cite
 * the research papers on the package. Check out http://www.gromacs.org.
 */
#include "gmxpre.h"

#include "detecthardware.h"

#include "config.h"

#include <cerrno>
#include <cstdlib>
#include <cstring>

#include <algorithm>
#include <chrono>
#include <string>
#include <thread>
#include <vector>

#include "thread_mpi/threads.h"

#include "gromacs/gmxlib/network.h"
#include "gromacs/gpu_utils/gpu_utils.h"
#include "gromacs/hardware/cpuinfo.h"
#include "gromacs/hardware/gpu_hw_info.h"
#include "gromacs/hardware/hardwareassign.h"
#include "gromacs/hardware/hardwaretopology.h"
#include "gromacs/hardware/hw_info.h"
#include "gromacs/mdtypes/commrec.h"
#include "gromacs/mdtypes/md_enums.h"
#include "gromacs/simd/support.h"
#include "gromacs/utility/arrayref.h"
#include "gromacs/utility/basedefinitions.h"
#include "gromacs/utility/basenetwork.h"
#include "gromacs/utility/baseversion.h"
#include "gromacs/utility/cstringutil.h"
#include "gromacs/utility/exceptions.h"
#include "gromacs/utility/fatalerror.h"
#include "gromacs/utility/gmxassert.h"
#include "gromacs/utility/logger.h"
#include "gromacs/utility/programcontext.h"
#include "gromacs/utility/smalloc.h"
#include "gromacs/utility/stringutil.h"
#include "gromacs/utility/sysinfo.h"

#ifdef HAVE_UNISTD_H
#    include <unistd.h>       // sysconf()
#endif

//! Convenience macro to help us avoid ifdefs each time we use sysconf
#if !defined(_SC_NPROCESSORS_ONLN) && defined(_SC_NPROC_ONLN)
#    define _SC_NPROCESSORS_ONLN _SC_NPROC_ONLN
#endif

//! Convenience macro to help us avoid ifdefs each time we use sysconf
#if !defined(_SC_NPROCESSORS_CONF) && defined(_SC_NPROC_CONF)
#    define _SC_NPROCESSORS_CONF _SC_NPROC_CONF
#endif

#if defined (__i386__) || defined (__x86_64__) || defined (_M_IX86) || defined (_M_X64)
//! Constant used to help minimize preprocessed code
static const bool isX86 = true;
#else
//! Constant used to help minimize preprocessed code
static const bool isX86 = false;
#endif

#if defined __powerpc__ || defined __ppc__ || defined __PPC__
static const bool isPowerPC = true;
#else
static const bool isPowerPC = false;
#endif

//! Constant used to help minimize preprocessed code
static const bool bGPUBinary     = GMX_GPU != GMX_GPU_NONE;

/* Note that some of the following arrays must match the "GPU support
 * enumeration" in src/config.h.cmakein, so that GMX_GPU looks up an
 * array entry. */

/* Both CUDA and OpenCL (on the supported/tested platforms) supports
 * GPU device sharing.
 */
static const bool gpuSharingSupport[] = { false, true, true };
static const bool bGpuSharingSupported = gpuSharingSupport[GMX_GPU];

/* Both CUDA and OpenCL (on the tested/supported platforms) supports everything.
 */
static const bool multiGpuSupport[] = {
    false, true, true
};
static const bool bMultiGpuPerNodeSupported = multiGpuSupport[GMX_GPU];

// TODO If/when we unify CUDA and OpenCL support code, this should
// move to a single place in gpu_utils.
/* Names of the GPU detection/check results (see e_gpu_detect_res_t in hw_info.h). */
const char * const gpu_detect_res_str[egpuNR] =
{
    "compatible", "inexistent", "incompatible", "insane"
};

static const char * invalid_gpuid_hint =
    "A delimiter-free sequence of valid numeric IDs of available GPUs is expected.";

/* The globally shared hwinfo structure. */
static gmx_hw_info_t      *hwinfo_g;
/* A reference counter for the hwinfo structure */
static int                 n_hwinfo = 0;
/* A lock to protect the hwinfo structure */
static tMPI_Thread_mutex_t hw_info_lock = TMPI_THREAD_MUTEX_INITIALIZER;

#define HOSTNAMELEN 80

/*! \brief Count and return the number of unique items in the container. */
template<typename T> inline size_t countUniqueItems(const std::vector<T> &store)
{
    std::set<T> uniqueStore;
    for (const auto &i : store)
    {
        uniqueStore.insert(i);
    }
    return uniqueStore.size();
}

gmx_bool gmx_multiple_gpu_per_node_supported()
{
    return bMultiGpuPerNodeSupported;
}

gmx_bool gmx_gpu_sharing_supported()
{
    return bGpuSharingSupported;
}

std::string sprint_gpus(const gmx_gpu_info_t *gpu_info)
{
    char                     stmp[STRLEN];
    std::vector<std::string> gpuStrings;
    for (int i = 0; i < gpu_info->n_dev; i++)
    {
        get_gpu_device_info_string(stmp, gpu_info, i);
        gpuStrings.push_back(gmx::formatString("    %s", stmp));
    }
    return gmx::joinStrings(gpuStrings, "\n");
}

/*! \brief Helper function for reporting GPU usage information
 * in the mdrun log file
 *
 * \param[in] gpu_info               Pointer to per-node GPU info struct
 * \param[in] gpu_opt                Pointer to per-node GPU options struct
 * \param[in] gpuTasks               GPU mapping information manager
 * \param[in] cr                     The communication structure
 * \param[out] gpusSharedAmongRanks  The boolean which tells if some ranks of the node use the same GPU
 * \return                           String to write to the log file (on the master rank)
 * \throws                   std::bad_alloc if out of memory */
static std::string
makeGpuUsageReport(const gmx_gpu_info_t *gpu_info,
                   const gmx_gpu_opt_t  *gpu_opt,
                   const GpuTaskManager &gpuTasks,
                   const t_commrec      *cr,
                   bool                 *gpusSharedAmongRanks)
{
    GMX_ASSERT(gpusSharedAmongRanks, "Pointer is needed");
    const int  ngpu_comp = gpu_info->n_dev_compatible;
    char       host[HOSTNAMELEN];

    const bool useMpi        = PAR(cr) || MULTISIM(cr);
    const bool printHostName = useMpi && GMX_LIB_MPI; // only when we actually have multiple hosts

    if (printHostName)
    {
        gmx_gethostname(host, HOSTNAMELEN);
    }

    /* First, collecting information about GPU task assignment */
    /* Preliminary reporting of GPU assignment (for each type of task - on the separate line) */
    std::string                          taskOutput;
    const std::map<GpuTask, std::string> gpuTasksToReport = {{GpuTask::NB, "PP"}, {GpuTask::PME, "PME"}};
    /* All the GPU IDs in use on the node will be in this container */
    std::vector<int>                     nodeGpuIds;
    /* This is a local GPU ID container for checking whether GPUs are being shared among ranks */
    std::vector<int>                     rankGpuIds;
    for (auto it : gpuTasksToReport)
    {
        const int invalidGpuId = -1;
        int       gpuId        = invalidGpuId;
        try
        {
            gpuId = gpuTasks.gpuId(it.first);
            rankGpuIds.push_back(gpuId);
        }
        catch (...)
        {
        }

        std::vector<int> nodeGpuTaskIds(cr->nrank_intranode, gpuId);
        if (useMpi)
        {
#if GMX_MPI
            // This gathers nodeTaskGpuIds from individual gpuIds on the master of the node
            MPI_Gather(&gpuId, 1, MPI_INT, nodeGpuTaskIds.data(), 1, MPI_INT, MASTERRANK(cr), cr->mpi_comm_physicalnode);
#endif
        }
        else
        {
            GMX_ASSERT(nodeGpuTaskIds.size() == 1, "This should be a serial case");
            nodeGpuTaskIds[0] = gpuId;
        }
        // Throwing out all the entries for ranks which don't run this GPU task
        nodeGpuTaskIds.erase(remove(nodeGpuTaskIds.begin(), nodeGpuTaskIds.end(), invalidGpuId), nodeGpuTaskIds.end());

        if (!nodeGpuTaskIds.empty())
        {
            const std::string nodeGpuIdsString = formatAndJoin(nodeGpuTaskIds, ",", gmx::StringFormatter("%d"));
            const bool        pluralTaskGpuIds = countUniqueItems<int>(nodeGpuTaskIds) > 1;
            const size_t      taskRanksCount   = nodeGpuTaskIds.size();

            taskOutput += gmx::formatString("Mapping of GPU ID%s to the %zu %s rank%s in this node: %s\n",
                                            pluralTaskGpuIds ? "s" : "",
                                            taskRanksCount,
                                            it.second.c_str(),
                                            (taskRanksCount > 1) ? "s" : "",
                                            nodeGpuIdsString.c_str());
        }
        nodeGpuIds.insert(nodeGpuIds.end(), nodeGpuTaskIds.begin(), nodeGpuTaskIds.end());
    }
    const size_t gpusInUseNode = countUniqueItems<int>(nodeGpuIds);
    *gpusSharedAmongRanks = false;
    if (useMpi)
    {
        unsigned int gpusInUseRank        = countUniqueItems<int>(rankGpuIds);
        unsigned int gpusInUseNodeReduced = gpusInUseRank;
#if GMX_MPI
        // This sums up gpusInUseRank into gpusInUseNodeReduced on all the ranks of the node
        MPI_Allreduce(&gpusInUseRank, &gpusInUseNodeReduced, 1, MPI_UNSIGNED, MPI_SUM, cr->mpi_comm_physicalnode);
#endif
        *gpusSharedAmongRanks = (gpusInUseNodeReduced != gpusInUseNode);
    }

    std::string  output;
    if ((ngpu_comp > 0) && (gpusInUseNode == 0))
    {
        /* Issue a single note if GPUs are available but not used */
        output = gmx::formatString("%d compatible GPU%s detected in the system, but none will be used.\n"
                                   "Consider trying GPU acceleration with the Verlet scheme!\n",
                                   ngpu_comp, (ngpu_comp > 1) ? "s" : "");
    }
    else
    {
        if (!gpu_opt->bUserSet)
        {
            /* Reporting compatible GPUs - gpu_opt->dev_compatible is only populated during auto-selection */
            std::string gpuIdsString =
                formatAndJoin(gmx::constArrayRefFromArray(gpu_opt->dev_compatible,
                                                          gpu_opt->n_dev_compatible),
                              ",", gmx::StringFormatter("%d"));
            bool bPluralGpus = gpu_opt->n_dev_compatible > 1;

            if (printHostName)
            {
                output += gmx::formatString("On host %s ", host);
            }
            output += gmx::formatString("%d compatible GPU%s %s present, with ID%s %s\n",
                                        gpu_opt->n_dev_compatible,
                                        bPluralGpus ? "s" : "",
                                        bPluralGpus ? "are" : "is",
                                        bPluralGpus ? "s" : "",
                                        gpuIdsString.c_str());
        }

        /* Reporting GPU task assignment */
        if (printHostName)
        {
            output += gmx::formatString("On host %s ", host);
        }

        output += gmx::formatString("%zu GPU task%s %sselected for this run, with %zu GPU%s.\n",
                                    nodeGpuIds.size(), (nodeGpuIds.size() > 1) ? "s" : "", gpu_opt->bUserSet ? "user-" : "auto-",
                                    gpusInUseNode, (gpusInUseNode > 1) ? "s" : "");
        output += taskOutput;
    }
    return output;
}

/* Give a suitable fatal error or warning if the build configuration
   and runtime CPU do not match. */
static void
check_use_of_rdtscp_on_this_cpu(const gmx::MDLogger   &mdlog,
                                const gmx::CpuInfo    &cpuInfo)
{
#ifdef HAVE_RDTSCP
    bool binaryUsesRdtscp = TRUE;
#else
    bool binaryUsesRdtscp = FALSE;
#endif

    const char *programName = gmx::getProgramContext().displayName();

    if (cpuInfo.supportLevel() < gmx::CpuInfo::SupportLevel::Features)
    {
        if (binaryUsesRdtscp)
        {
            GMX_LOG(mdlog.warning).asParagraph().appendTextFormatted(
                    "The %s executable was compiled to use the rdtscp CPU instruction. "
                    "We cannot detect the features of your current CPU, but will proceed anyway. "
                    "If you get a crash, rebuild GROMACS with the GMX_USE_RDTSCP=OFF CMake option.",
                    programName);
        }
    }
    else
    {
        bool cpuHasRdtscp = cpuInfo.feature(gmx::CpuInfo::Feature::X86_Rdtscp);

        if (!cpuHasRdtscp && binaryUsesRdtscp)
        {
            gmx_fatal(FARGS, "The %s executable was compiled to use the rdtscp CPU instruction. "
                      "However, this is not supported by the current hardware and continuing would lead to a crash. "
                      "Please rebuild GROMACS with the GMX_USE_RDTSCP=OFF CMake option.",
                      programName);
        }

        if (cpuHasRdtscp && !binaryUsesRdtscp)
        {
            GMX_LOG(mdlog.warning).asParagraph().appendTextFormatted(
                    "The current CPU can measure timings more accurately than the code in\n"
                    "%s was configured to use. This might affect your simulation\n"
                    "speed as accurate timings are needed for load-balancing.\n"
                    "Please consider rebuilding %s with the GMX_USE_RDTSCP=ON CMake option.",
                    programName, programName);
        }
    }
}

void gmx_check_hw_runconf_consistency(const gmx::MDLogger  &mdlog,
                                      const gmx_hw_info_t  *hwinfo,
                                      const t_commrec      *cr,
                                      const gmx_hw_opt_t   *hw_opt,
                                      gmx_bool              bUseGPU,
                                      const GpuTaskManager &gpuTasks)
{
    gmx_bool btMPI, bMPI, bNthreadsAuto, bEmulateGPU;

    GMX_RELEASE_ASSERT(hwinfo, "hwinfo must be a non-NULL pointer");
    GMX_RELEASE_ASSERT(cr, "cr must be a non-NULL pointer");

#if GMX_THREAD_MPI
    bMPI          = FALSE;
    btMPI         = TRUE;
    bNthreadsAuto = (hw_opt->nthreads_tmpi < 1);
#elif GMX_LIB_MPI
    bMPI          = TRUE;
    btMPI         = FALSE;
    bNthreadsAuto = FALSE;
#else
    bMPI          = FALSE;
    btMPI         = FALSE;
    bNthreadsAuto = FALSE;
#endif

    /* GPU emulation detection is done later, but we need here as well
     * -- uncool, but there's no elegant workaround */
    bEmulateGPU       = (getenv("GMX_EMULATE_GPU") != nullptr);
    bool gpusSharedAmongRanks = false;

    if (hwinfo->gpu_info.n_dev_compatible > 0)
    {
        std::string gpuUsageReport;
        try
        {
            gpuUsageReport = makeGpuUsageReport(&hwinfo->gpu_info,
                                                &hw_opt->gpu_opt,
                                                gpuTasks,
                                                cr,
                                                &gpusSharedAmongRanks);
        }
        GMX_CATCH_ALL_AND_EXIT_WITH_FATAL_ERROR;

        /* NOTE: this print is only for and on one physical node */
        GMX_LOG(mdlog.warning).appendText(gpuUsageReport);
    }

    /* Need to ensure that we have enough GPUs:
     * - need one GPU per any GPU-using rank
     * - no GPU oversubscription with tMPI
     */

    int        gpuTasksCount    = gpuTasks.rankGpuTasksCount();
    int        gpuRankIndicator = (gpuTasksCount > 0) ? 1 : 0;
    /* Default values for serial case */
    int        gpuTasksNodeCount = gpuTasksCount;
    int        gpuRanksNodeCount = gpuRankIndicator;
    const bool useMpi            = PAR(cr) || MULTISIM(cr);
    if (useMpi)
    {
#if GMX_MPI
        // This sums up gpuRankIndicator into gpuRanksNodeCount on all the ranks of the node
        MPI_Allreduce(&gpuRankIndicator, &gpuRanksNodeCount, 1, MPI_INT, MPI_SUM, cr->mpi_comm_physicalnode);
        // This sums up gpuTasksCount into gpuTasksNodeCount on all the ranks of the node
        MPI_Allreduce(&gpuTasksCount, &gpuTasksNodeCount, 1, MPI_INT, MPI_SUM, cr->mpi_comm_physicalnode);
#endif
    }

    // The following checks only concern GPU-using ranks
    if (!gpuRankIndicator)
    {
        return;
    }

    /* Default strings for neither MPI nor tMPI (serial case) */
    std::string rankString  = "process", pernode;
    std::string ranksString = rankString;
    if (btMPI)
    {
        rankString  = "thread-MPI thread";
        ranksString = rankString + ((gpuRanksNodeCount > 1) ? "s" : "");
    }
    else if (bMPI)
    {
        rankString  = "MPI process";
        ranksString = rankString + ((gpuRanksNodeCount > 1) ? "es" : "");
        pernode     = " per node";
    }

    if (bUseGPU && hwinfo->gpu_info.n_dev_compatible > 0 &&
        !bEmulateGPU)
    {
        const int         ngpu_comp            = hwinfo->gpu_info.n_dev_compatible;
        const std::string gpuTasksPlural       = (gpuTasksNodeCount > 1) ? "s" : "";
        const std::string gpusCompatiblePlural = (ngpu_comp > 1) ? "s" : "";
        const char       *programName          = gmx::getProgramContext().displayName();

        /* number of tMPI threads was auto-adjusted */
        if (btMPI && bNthreadsAuto)
        {
            if (hw_opt->gpu_opt.bUserSet && (gpuTasksNodeCount < hw_opt->gpu_opt.n_dev_use))
            {
                /* The user manually provided more GPUs than threads we could automatically start. */
                gmx_fatal(FARGS,
                          "%d GPU ID%s provided, but only %d GPU-using %s\n"
                          "with %d GPU tasks in total could be started.\n"
                          "Please use fewer GPU IDs.",
                          hw_opt->gpu_opt.n_dev_use, (hw_opt->gpu_opt.n_dev_use > 1) ? "s" : "",
                          gpuRanksNodeCount, ranksString.c_str(), gpuTasksNodeCount);
            }

            if (!hw_opt->gpu_opt.bUserSet && (gpuTasksNodeCount < ngpu_comp))
            {
                /* There are more GPUs than GPU tasks on the node; we have limited the number of GPUs used. */
                GMX_LOG(mdlog.warning).asParagraph().appendTextFormatted(
                        "NOTE: %d GPU%s were detected, but only %d GPU-using %s\n"
                        "      with %d GPU tasks in total can be started.\n"
                        "      Only %d GPU%s will be used by %s.",
                        ngpu_comp, gpusCompatiblePlural.c_str(),
                        gpuRanksNodeCount, ranksString.c_str(),
                        gpuTasksNodeCount, gpuTasksPlural.c_str(), programName);
            }
        }

        /* Auto-selected GPUs check;
         * user-selected GPu IDs have also been checked in GpuTaskAssignmentManager::selectRankGpus
         */
        if (!hw_opt->gpu_opt.bUserSet)
        {
            /* TODO Should we have a gpu_opt->n_dev_supported field? */
            if ((ngpu_comp > gpuTasksNodeCount) && gmx_multiple_gpu_per_node_supported())
            {
                /* There is not enough GPU tasks to use all the GPUs */
                GMX_LOG(mdlog.warning).asParagraph().appendTextFormatted(
                        "NOTE: potentially sub-optimal launch configuration, %s started with less\n"
                        "      GPU tasks%s than GPU%s available. %d GPU%s%s will be used.",
                        programName, pernode.c_str(), gpusCompatiblePlural.c_str(),
                        gpuTasksNodeCount, gpuTasksPlural.c_str(), pernode.c_str());
            }

            /* GPUs were assigned automatically, but number of GPU-using ranks somehow doesn't correspond.
             * TODO: Can this error be even triggered nowadays? (Instead of triggering "non-multiple GPU count" error).
             * Probably should be removed for multiple GPU tasks per rank.
             */
#ifdef FIXME
            if (gpuTasksNodeCount != gpuRanksNodeCount)
            {
                /* Avoid duplicate error messages.
                 * Unfortunately we can only do this at the physical node
                 * level, since the hardware setup and MPI process count
                 * might differ between physical nodes.
                 */
                const bool handleNodeError = (cr->rank_pp_intranode == 0); // This should be the only rank on the node
                if (handleNodeError)
                {
                    std::string reasonForLimit;
                    if (ngpu_comp > 1 &&
                        gpuTasksNodeCount == 1 &&
                        !gmx_multiple_gpu_per_node_supported()) // TODO remove gmx_multiple_gpu_per_node_supported() altogether?
                    {
                        reasonForLimit  = "can be used by ";
                        reasonForLimit += getGpuImplementationString();
                        reasonForLimit += " in GROMACS";
                    }
                    else
                    {
                        reasonForLimit = "was detected";
                    }
                    gmx_fatal(FARGS,
                              "Incorrect launch configuration: mismatching number of GPU-using %s and GPUs%s.\n"
                              "%s was started with %d GPU-using %s%s, but only %d GPU%s %s.",
                              ranksString.c_str(), pernode.c_str(),
                              programName, gpuRanksNodeCount, ranksString.c_str(), pernode.c_str(),
                              gpuTasksNodeCount, gpuTasksPlural.c_str(), reasonForLimit.c_str()); // TODO count unique GPU ids here as well?
                }
            }
#endif
        }
        /* Some ranks might share a GPU, which generally degrades performance */
        // TODO: should this be internal to GpuTaskAssignmentManager::selectRankGpus?
        if (gpusSharedAmongRanks)
        {
            GMX_LOG(mdlog.warning).appendTextFormatted("NOTE: You have assigned some GPUs to multiple %s.", ranksString.c_str());
        }
    }

#if GMX_MPI
    if (PAR(cr))
    {
        /* Avoid other ranks to continue after
           inconsistency */
        MPI_Barrier(cr->mpi_comm_mygroup);
    }
#endif
}

static void gmx_detect_gpus(const gmx::MDLogger &mdlog, const t_commrec *cr)
{
#if GMX_LIB_MPI
    int              rank_world;
    MPI_Comm         physicalnode_comm;
#endif
    int              rank_local;

    /* Under certain circumstances MPI ranks on the same physical node
     * can not simultaneously access the same GPU(s). Therefore we run
     * the detection only on one MPI rank per node and broadcast the info.
     * Note that with thread-MPI only a single thread runs this code.
     *
     * NOTE: We can't broadcast gpu_info with OpenCL as the device and platform
     * ID stored in the structure are unique for each rank (even if a device
     * is shared by multiple ranks).
     *
     * TODO: We should also do CPU hardware detection only once on each
     * physical node and broadcast it, instead of do it on every MPI rank.
     */
#if GMX_LIB_MPI
    /* A split of MPI_COMM_WORLD over physical nodes is only required here,
     * so we create and destroy it locally.
     */
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_world);
    MPI_Comm_split(MPI_COMM_WORLD, gmx_physicalnode_id_hash(),
                   rank_world, &physicalnode_comm);
    MPI_Comm_rank(physicalnode_comm, &rank_local);
    GMX_UNUSED_VALUE(cr);
#else
    /* Here there should be only one process, check this */
    GMX_RELEASE_ASSERT(cr->nnodes == 1 && cr->sim_nodeid == 0, "Only a single (master) process should execute here");

    rank_local = 0;
#endif

    /*  With CUDA detect only on one rank per host, with OpenCL need do
     *  the detection on all PP ranks */
    bool isOpenclPpRank = ((GMX_GPU == GMX_GPU_OPENCL) && (cr->duty & DUTY_PP));

    if (rank_local == 0 || isOpenclPpRank)
    {
        char detection_error[STRLEN] = "", sbuf[STRLEN];

        if (detect_gpus(&hwinfo_g->gpu_info, detection_error) != 0)
        {
            if (detection_error[0] != '\0')
            {
                sprintf(sbuf, ":\n      %s\n", detection_error);
            }
            else
            {
                sprintf(sbuf, ".");
            }
            GMX_LOG(mdlog.warning).asParagraph().appendTextFormatted(
                    "NOTE: Error occurred during GPU detection%s"
                    "      Can not use GPU acceleration, will fall back to CPU kernels.",
                    sbuf);
        }
    }

#if GMX_LIB_MPI
    if (!isOpenclPpRank)
    {
        /* Broadcast the GPU info to the other ranks within this node */
        MPI_Bcast(&hwinfo_g->gpu_info.n_dev, 1, MPI_INT, 0, physicalnode_comm);

        if (hwinfo_g->gpu_info.n_dev > 0)
        {
            int dev_size;

            dev_size = hwinfo_g->gpu_info.n_dev*sizeof_gpu_dev_info();

            if (rank_local > 0)
            {
                hwinfo_g->gpu_info.gpu_dev =
                    (struct gmx_device_info_t *)malloc(dev_size);
            }
            MPI_Bcast(hwinfo_g->gpu_info.gpu_dev, dev_size, MPI_BYTE,
                      0, physicalnode_comm);
            MPI_Bcast(&hwinfo_g->gpu_info.n_dev_compatible, 1, MPI_INT,
                      0, physicalnode_comm);
        }
    }

    MPI_Comm_free(&physicalnode_comm);
#endif
}

static void gmx_collect_hardware_mpi(const gmx::CpuInfo &cpuInfo)
{
    const int ncore = hwinfo_g->hardwareTopology->numberOfCores();
#if GMX_LIB_MPI
    int       rank_id;
    int       nrank, rank, nhwthread, ngpu, i;
    int       gpu_hash;
    int      *buf, *all;

    rank_id   = gmx_physicalnode_id_hash();
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nrank);
    nhwthread = hwinfo_g->nthreads_hw_avail;
    ngpu      = hwinfo_g->gpu_info.n_dev_compatible;
    /* Create a unique hash of the GPU type(s) in this node */
    gpu_hash  = 0;
    /* Here it might be better to only loop over the compatible GPU, but we
     * don't have that information available and it would also require
     * removing the device ID from the device info string.
     */
    for (i = 0; i < hwinfo_g->gpu_info.n_dev; i++)
    {
        char stmp[STRLEN];

        /* Since the device ID is incorporated in the hash, the order of
         * the GPUs affects the hash. Also two identical GPUs won't give
         * a gpu_hash of zero after XORing.
         */
        get_gpu_device_info_string(stmp, &hwinfo_g->gpu_info, i);
        gpu_hash ^= gmx_string_fullhash_func(stmp, gmx_string_hash_init);
    }

    snew(buf, nrank);
    snew(all, nrank);
    buf[rank] = rank_id;

    MPI_Allreduce(buf, all, nrank, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    gmx_bool bFound;
    int      nnode0, ncore0, nhwthread0, ngpu0, r;

    bFound     = FALSE;
    ncore0     = 0;
    nnode0     = 0;
    nhwthread0 = 0;
    ngpu0      = 0;
    for (r = 0; r < nrank; r++)
    {
        if (all[r] == rank_id)
        {
            if (!bFound && r == rank)
            {
                /* We are the first rank in this physical node */
                nnode0     = 1;
                ncore0     = ncore;
                nhwthread0 = nhwthread;
                ngpu0      = ngpu;
            }
            bFound = TRUE;
        }
    }

    sfree(buf);
    sfree(all);

    int sum[4], maxmin[10];

    {
        int buf[4];

        /* Sum values from only intra-rank 0 so we get the sum over all nodes */
        buf[0] = nnode0;
        buf[1] = ncore0;
        buf[2] = nhwthread0;
        buf[3] = ngpu0;

        MPI_Allreduce(buf, sum, 4, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    }

    {
        int buf[10];

        /* Store + and - values for all ranks,
         * so we can get max+min with one MPI call.
         */
        buf[0] = ncore;
        buf[1] = nhwthread;
        buf[2] = ngpu;
        buf[3] = static_cast<int>(gmx::simdSuggested(cpuInfo));
        buf[4] = gpu_hash;
        buf[5] = -buf[0];
        buf[6] = -buf[1];
        buf[7] = -buf[2];
        buf[8] = -buf[3];
        buf[9] = -buf[4];

        MPI_Allreduce(buf, maxmin, 10, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    }

    hwinfo_g->nphysicalnode       = sum[0];
    hwinfo_g->ncore_tot           = sum[1];
    hwinfo_g->ncore_min           = -maxmin[5];
    hwinfo_g->ncore_max           = maxmin[0];
    hwinfo_g->nhwthread_tot       = sum[2];
    hwinfo_g->nhwthread_min       = -maxmin[6];
    hwinfo_g->nhwthread_max       = maxmin[1];
    hwinfo_g->ngpu_compatible_tot = sum[3];
    hwinfo_g->ngpu_compatible_min = -maxmin[7];
    hwinfo_g->ngpu_compatible_max = maxmin[2];
    hwinfo_g->simd_suggest_min    = -maxmin[8];
    hwinfo_g->simd_suggest_max    = maxmin[3];
    hwinfo_g->bIdenticalGPUs      = (maxmin[4] == -maxmin[9]);
#else
    /* All ranks use the same pointer, protected by a mutex in the caller */
    hwinfo_g->nphysicalnode       = 1;
    hwinfo_g->ncore_tot           = ncore;
    hwinfo_g->ncore_min           = ncore;
    hwinfo_g->ncore_max           = ncore;
    hwinfo_g->nhwthread_tot       = hwinfo_g->nthreads_hw_avail;
    hwinfo_g->nhwthread_min       = hwinfo_g->nthreads_hw_avail;
    hwinfo_g->nhwthread_max       = hwinfo_g->nthreads_hw_avail;
    hwinfo_g->ngpu_compatible_tot = hwinfo_g->gpu_info.n_dev_compatible;
    hwinfo_g->ngpu_compatible_min = hwinfo_g->gpu_info.n_dev_compatible;
    hwinfo_g->ngpu_compatible_max = hwinfo_g->gpu_info.n_dev_compatible;
    hwinfo_g->simd_suggest_min    = static_cast<int>(simdSuggested(cpuInfo));
    hwinfo_g->simd_suggest_max    = static_cast<int>(simdSuggested(cpuInfo));
    hwinfo_g->bIdenticalGPUs      = TRUE;
#endif
}

/*! \brief Utility that does dummy computing for max 2 seconds to spin up cores
 *
 *  This routine will check the number of cores configured and online
 *  (using sysconf), and the spins doing dummy compute operations for up to
 *  2 seconds, or until all cores have come online. This can be used prior to
 *  hardware detection for platforms that take unused processors offline.
 *
 *  This routine will not throw exceptions.
 */
static void
spinUpCore() noexcept
{
#if defined(HAVE_SYSCONF) && defined(_SC_NPROCESSORS_CONF) && defined(_SC_NPROCESSORS_ONLN)
    float dummy           = 0.1;
    int   countConfigured = sysconf(_SC_NPROCESSORS_CONF);    // noexcept
    auto  start           = std::chrono::steady_clock::now(); // noexcept

    while (sysconf(_SC_NPROCESSORS_ONLN) < countConfigured &&
           std::chrono::steady_clock::now() - start < std::chrono::seconds(2))
    {
        for (int i = 1; i < 10000; i++)
        {
            dummy /= i;
        }
    }

    if (dummy < 0)
    {
        printf("This cannot happen, but prevents loop from being optimized away.");
    }
#endif
}

/*! \brief Prepare the system before hardware topology detection
 *
 * This routine should perform any actions we want to put the system in a state
 * where we want it to be before detecting the hardware topology. For most
 * processors there is nothing to do, but some architectures (in particular ARM)
 * have support for taking configured cores offline, which will make them disappear
 * from the online processor count.
 *
 * This routine checks if there is a mismatch between the number of cores
 * configured and online, and in that case we issue a small workload that
 * attempts to wake sleeping cores before doing the actual detection.
 *
 * This type of mismatch can also occur for x86 or PowerPC on Linux, if SMT has only
 * been disabled in the kernel (rather than bios). Since those cores will never
 * come online automatically, we currently skip this test for x86 & PowerPC to
 * avoid wasting 2 seconds. We also skip the test if there is no thread support.
 *
 * \note Cores will sleep relatively quickly again, so it's important to issue
 *       the real detection code directly after this routine.
 */
static void
hardwareTopologyPrepareDetection()
{
#if defined(HAVE_SYSCONF) && defined(_SC_NPROCESSORS_CONF) && \
    (defined(THREAD_PTHREADS) || defined(THREAD_WINDOWS))

    // Modify this conditional when/if x86 or PowerPC starts to sleep some cores
    if (!isX86 && !isPowerPC)
    {
        int                      countConfigured  = sysconf(_SC_NPROCESSORS_CONF);
        std::vector<std::thread> workThreads(countConfigured);

        for (auto &t : workThreads)
        {
            t = std::thread(spinUpCore);
        }

        for (auto &t : workThreads)
        {
            t.join();
        }
    }
#endif
}

/*! \brief Sanity check hardware topology and print some notes to log
 *
 *  \param mdlog            Logger.
 *  \param hardwareTopology Reference to hardwareTopology object.
 */
static void
hardwareTopologyDoubleCheckDetection(const gmx::MDLogger gmx_unused         &mdlog,
                                     const gmx::HardwareTopology gmx_unused &hardwareTopology)
{
#if defined HAVE_SYSCONF && defined(_SC_NPROCESSORS_CONF)
    if (hardwareTopology.supportLevel() < gmx::HardwareTopology::SupportLevel::LogicalProcessorCount)
    {
        return;
    }

    int countFromDetection = hardwareTopology.machine().logicalProcessorCount;
    int countConfigured    = sysconf(_SC_NPROCESSORS_CONF);

    /* BIOS, kernel or user actions can take physical processors
     * offline. We already cater for the some of the cases inside the hardwareToplogy
     * by trying to spin up cores just before we detect, but there could be other
     * cases where it is worthwhile to hint that there might be more resources available.
     */
    if (countConfigured >= 0 && countConfigured != countFromDetection)
    {
        GMX_LOG(mdlog.info).
            appendTextFormatted("Note: %d CPUs configured, but only %d were detected to be online.\n", countConfigured, countFromDetection);

        if (isX86 && countConfigured == 2*countFromDetection)
        {
            GMX_LOG(mdlog.info).
                appendText("      X86 Hyperthreading is likely disabled; enable it for better performance.");
        }
        // For PowerPC (likely Power8) it is possible to set SMT to either 2,4, or 8-way hardware threads.
        // We only warn if it is completely disabled since default performance drops with SMT8.
        if (isPowerPC && countConfigured == 8*countFromDetection)
        {
            GMX_LOG(mdlog.info).
                appendText("      PowerPC SMT is likely disabled; enable SMT2/SMT4 for better performance.");
        }
    }
#endif
}


gmx_hw_info_t *gmx_detect_hardware(const gmx::MDLogger &mdlog, const t_commrec *cr,
                                   gmx_bool bDetectGPUs)
{
    int ret;

    /* make sure no one else is doing the same thing */
    ret = tMPI_Thread_mutex_lock(&hw_info_lock);
    if (ret != 0)
    {
        gmx_fatal(FARGS, "Error locking hwinfo mutex: %s", strerror(errno));
    }

    /* only initialize the hwinfo structure if it is not already initalized */
    if (n_hwinfo == 0)
    {
        snew(hwinfo_g, 1);

        hwinfo_g->cpuInfo             = new gmx::CpuInfo(gmx::CpuInfo::detect());

        hardwareTopologyPrepareDetection();
        hwinfo_g->hardwareTopology    = new gmx::HardwareTopology(gmx::HardwareTopology::detect());

        // If we detected the topology on this system, double-check that it makes sense
        if (hwinfo_g->hardwareTopology->isThisSystem())
        {
            hardwareTopologyDoubleCheckDetection(mdlog, *(hwinfo_g->hardwareTopology));
        }

        // TODO: Get rid of this altogether.
        hwinfo_g->nthreads_hw_avail = hwinfo_g->hardwareTopology->machine().logicalProcessorCount;

        /* detect GPUs */
        hwinfo_g->gpu_info.n_dev            = 0;
        hwinfo_g->gpu_info.n_dev_compatible = 0;
        hwinfo_g->gpu_info.gpu_dev          = nullptr;

        /* Run the detection if the binary was compiled with GPU support
         * and we requested detection.
         */
        hwinfo_g->gpu_info.bDetectGPUs =
            (bGPUBinary && bDetectGPUs &&
             getenv("GMX_DISABLE_GPU_DETECTION") == nullptr);
        if (hwinfo_g->gpu_info.bDetectGPUs)
        {
            gmx_detect_gpus(mdlog, cr);
        }

        gmx_collect_hardware_mpi(*hwinfo_g->cpuInfo);
    }
    /* increase the reference counter */
    n_hwinfo++;

    ret = tMPI_Thread_mutex_unlock(&hw_info_lock);
    if (ret != 0)
    {
        gmx_fatal(FARGS, "Error unlocking hwinfo mutex: %s", strerror(errno));
    }

    return hwinfo_g;
}

static std::string detected_hardware_string(const gmx_hw_info_t *hwinfo,
                                            bool                 bFullCpuInfo)
{
    std::string                  s;

    const gmx::CpuInfo          &cpuInfo = *hwinfo_g->cpuInfo;
    const gmx::HardwareTopology &hwTop   = *hwinfo->hardwareTopology;

    s  = gmx::formatString("\n");
    s += gmx::formatString("Running on %d node%s with total",
                           hwinfo->nphysicalnode,
                           hwinfo->nphysicalnode == 1 ? "" : "s");
    if (hwinfo->ncore_tot > 0)
    {
        s += gmx::formatString(" %d cores,", hwinfo->ncore_tot);
    }
    s += gmx::formatString(" %d logical cores", hwinfo->nhwthread_tot);
    if (hwinfo->gpu_info.bDetectGPUs)
    {
        s += gmx::formatString(", %d compatible GPU%s",
                               hwinfo->ngpu_compatible_tot,
                               hwinfo->ngpu_compatible_tot == 1 ? "" : "s");
    }
    else if (bGPUBinary)
    {
        s += gmx::formatString(" (GPU detection deactivated)");
    }
    s += gmx::formatString("\n");

    if (hwinfo->nphysicalnode > 1)
    {
        /* Print per node hardware feature counts */
        if (hwinfo->ncore_max > 0)
        {
            s += gmx::formatString("  Cores per node:           %2d", hwinfo->ncore_min);
            if (hwinfo->ncore_max > hwinfo->ncore_min)
            {
                s += gmx::formatString(" - %2d", hwinfo->ncore_max);
            }
            s += gmx::formatString("\n");
        }
        s += gmx::formatString("  Logical cores per node:   %2d", hwinfo->nhwthread_min);
        if (hwinfo->nhwthread_max > hwinfo->nhwthread_min)
        {
            s += gmx::formatString(" - %2d", hwinfo->nhwthread_max);
        }
        s += gmx::formatString("\n");
        if (bGPUBinary)
        {
            s += gmx::formatString("  Compatible GPUs per node: %2d",
                                   hwinfo->ngpu_compatible_min);
            if (hwinfo->ngpu_compatible_max > hwinfo->ngpu_compatible_min)
            {
                s += gmx::formatString(" - %2d", hwinfo->ngpu_compatible_max);
            }
            s += gmx::formatString("\n");
            if (hwinfo->ngpu_compatible_tot > 0)
            {
                if (hwinfo->bIdenticalGPUs)
                {
                    s += gmx::formatString("  All nodes have identical type(s) of GPUs\n");
                }
                else
                {
                    /* This message will also appear with identical GPU types
                     * when at least one node has no GPU.
                     */
                    s += gmx::formatString("  Different nodes have different type(s) and/or order of GPUs\n");
                }
            }
        }
    }

#if GMX_LIB_MPI
    char host[HOSTNAMELEN];
    int  rank;

    gmx_gethostname(host, HOSTNAMELEN);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    s += gmx::formatString("Hardware detected on host %s (the node of MPI rank %d):\n",
                           host, rank);
#else
    s += gmx::formatString("Hardware detected:\n");
#endif
    s += gmx::formatString("  CPU info:\n");

    s += gmx::formatString("    Vendor: %s\n", cpuInfo.vendorString().c_str());

    s += gmx::formatString("    Brand:  %s\n", cpuInfo.brandString().c_str());

    if (bFullCpuInfo)
    {
        s += gmx::formatString("    Family: %d   Model: %d   Stepping: %d\n",
                               cpuInfo.family(), cpuInfo.model(), cpuInfo.stepping());

        s += gmx::formatString("    Features:");
        for (auto &f : cpuInfo.featureSet())
        {
            s += gmx::formatString(" %s", cpuInfo.featureString(f).c_str());;
        }
        s += gmx::formatString("\n");
    }

    s += gmx::formatString("    SIMD instructions most likely to fit this hardware: %s",
                           gmx::simdString(static_cast<gmx::SimdType>(hwinfo->simd_suggest_min)).c_str());

    if (hwinfo->simd_suggest_max > hwinfo->simd_suggest_min)
    {
        s += gmx::formatString(" - %s", gmx::simdString(static_cast<gmx::SimdType>(hwinfo->simd_suggest_max)).c_str());
    }
    s += gmx::formatString("\n");

    s += gmx::formatString("    SIMD instructions selected at GROMACS compile time: %s\n",
                           gmx::simdString(gmx::simdCompiled()).c_str());

    s += gmx::formatString("\n");

    s += gmx::formatString("  Hardware topology: ");
    switch (hwTop.supportLevel())
    {
        case gmx::HardwareTopology::SupportLevel::None:
            s += gmx::formatString("None\n");
            break;
        case gmx::HardwareTopology::SupportLevel::LogicalProcessorCount:
            s += gmx::formatString("Only logical processor count\n");
            break;
        case gmx::HardwareTopology::SupportLevel::Basic:
            s += gmx::formatString("Basic\n");
            break;
        case gmx::HardwareTopology::SupportLevel::Full:
            s += gmx::formatString("Full\n");
            break;
        case gmx::HardwareTopology::SupportLevel::FullWithDevices:
            s += gmx::formatString("Full, with devices\n");
            break;
    }

    if (!hwTop.isThisSystem())
    {
        s += gmx::formatString("  NOTE: Hardware topology cached or synthetic, not detected.\n");
        if (char *p = getenv("HWLOC_XMLFILE"))
        {
            s += gmx::formatString("        HWLOC_XMLFILE=%s\n", p);
        }
    }

    if (bFullCpuInfo)
    {
        if (hwTop.supportLevel() >= gmx::HardwareTopology::SupportLevel::Basic)
        {
            s += gmx::formatString("    Sockets, cores, and logical processors:\n");

            for (auto &socket : hwTop.machine().sockets)
            {
                s += gmx::formatString("      Socket %2d:", socket.id);
                for (auto &c : socket.cores)
                {
                    s += gmx::formatString(" [");
                    for (auto &t : c.hwThreads)
                    {
                        s += gmx::formatString(" %3d", t.logicalProcessorId);
                    }
                    s += gmx::formatString("]");
                }
                s += gmx::formatString("\n");
            }
        }
        if (hwTop.supportLevel() >= gmx::HardwareTopology::SupportLevel::Full)
        {
            s += gmx::formatString("    Numa nodes:\n");
            for (auto &n : hwTop.machine().numa.nodes)
            {
                s += gmx::formatString("      Node %2d (%" GMX_PRIu64 " bytes mem):", n.id, n.memory);
                for (auto &l : n.logicalProcessorId)
                {
                    s += gmx::formatString(" %3d", l);
                }
                s += gmx::formatString("\n");
            }
            s += gmx::formatString("      Latency:\n          ");
            for (std::size_t j = 0; j < hwTop.machine().numa.nodes.size(); j++)
            {
                s += gmx::formatString(" %5d", j);
            }
            s += gmx::formatString("\n");
            for (std::size_t i = 0; i < hwTop.machine().numa.nodes.size(); i++)
            {
                s += gmx::formatString("     %5d", i);
                for (std::size_t j = 0; j < hwTop.machine().numa.nodes.size(); j++)
                {
                    s += gmx::formatString(" %5.2f", hwTop.machine().numa.relativeLatency[i][j]);
                }
                s += gmx::formatString("\n");
            }


            s += gmx::formatString("    Caches:\n");
            for (auto &c : hwTop.machine().caches)
            {
                s += gmx::formatString("      L%d: %" GMX_PRIu64 " bytes, linesize %d bytes, assoc. %d, shared %d ways\n",
                                       c.level, c.size, c.linesize, c.associativity, c.shared);
            }
        }
        if (hwTop.supportLevel() >= gmx::HardwareTopology::SupportLevel::FullWithDevices)
        {
            s += gmx::formatString("    PCI devices:\n");
            for (auto &d : hwTop.machine().devices)
            {
                s += gmx::formatString("      %04x:%02x:%02x.%1x  Id: %04x:%04x  Class: 0x%04x  Numa: %d\n",
                                       d.domain, d.bus, d.dev, d.func, d.vendorId, d.deviceId, d.classId, d.numaNodeId);
            }
        }
    }

    if (bGPUBinary && (hwinfo->ngpu_compatible_tot > 0 ||
                       hwinfo->gpu_info.n_dev > 0))
    {
        s += gmx::formatString("  GPU info:\n");
        s += gmx::formatString("    Number of GPUs detected: %d\n",
                               hwinfo->gpu_info.n_dev);
        if (hwinfo->gpu_info.n_dev > 0)
        {
            s += sprint_gpus(&hwinfo->gpu_info) + "\n";
        }
    }
    return s;
}

void gmx_print_detected_hardware(FILE *fplog, const t_commrec *cr,
                                 const gmx::MDLogger &mdlog,
                                 const gmx_hw_info_t *hwinfo)
{
    const gmx::CpuInfo &cpuInfo = *hwinfo_g->cpuInfo;

    if (fplog != nullptr)
    {
        std::string detected;

        detected = detected_hardware_string(hwinfo, TRUE);

        fprintf(fplog, "%s\n", detected.c_str());
    }

    if (MULTIMASTER(cr))
    {
        std::string detected;

        detected = detected_hardware_string(hwinfo, FALSE);

        fprintf(stderr, "%s\n", detected.c_str());
    }

    /* Check the compiled SIMD instruction set against that of the node
     * with the lowest SIMD level support (skip if SIMD detection did not work)
     */
    if (cpuInfo.supportLevel() >= gmx::CpuInfo::SupportLevel::Features)
    {
        gmx::simdCheck(static_cast<gmx::SimdType>(hwinfo->simd_suggest_min), fplog, MULTIMASTER(cr));
    }

    /* For RDTSCP we only check on our local node and skip the MPI reduction */
    check_use_of_rdtscp_on_this_cpu(mdlog, cpuInfo);
}

//! \brief Return if any GPU ID (e.g in a user-supplied string) is repeated
static gmx_bool anyGpuIdIsRepeated(const gmx_gpu_opt_t *gpu_opt)
{
    /* Loop over IDs in the string */
    for (int i = 0; i < gpu_opt->n_dev_use - 1; ++i)
    {
        /* Look for the ID in location i in the following part of the
           string */
        for (int j = i + 1; j < gpu_opt->n_dev_use; ++j)
        {
            if (gpu_opt->dev_use[i] == gpu_opt->dev_use[j])
            {
                /* Same ID found in locations i and j */
                return TRUE;
            }
        }
    }

    return FALSE;
}

void gmx_parse_gpu_ids(gmx_gpu_opt_t *gpu_opt)
{
    char *env;

    if (gpu_opt->gpu_id != nullptr && !bGPUBinary)
    {
        gmx_fatal(FARGS, "GPU ID string set, but %s was compiled without GPU support!",
                  gmx::getProgramContext().displayName());
    }

    env = getenv("GMX_GPU_ID");
    if (env != nullptr && gpu_opt->gpu_id != nullptr)
    {
        gmx_fatal(FARGS, "GMX_GPU_ID and -gpu_id can not be used at the same time");
    }
    if (env == nullptr)
    {
        env = gpu_opt->gpu_id;
    }

    /* parse GPU IDs if the user passed any */
    if (env != nullptr)
    {
        /* Parse a "plain" or comma-separated GPU ID string which contains a
         * sequence of digits corresponding to GPU IDs; the order will
         * indicate the process/tMPI thread - GPU assignment. */
        parse_digits_from_string(env, &gpu_opt->n_dev_use, &gpu_opt->dev_use);

        if (!gmx_multiple_gpu_per_node_supported() && 1 < gpu_opt->n_dev_use)
        {
            gmx_fatal(FARGS, "The %s implementation only supports using exactly one PP rank per node", getGpuImplementationString());
        }
        if (!gmx_gpu_sharing_supported() && anyGpuIdIsRepeated(gpu_opt))
        {
            gmx_fatal(FARGS, "The %s implementation only supports using exactly one PP rank per GPU", getGpuImplementationString());
        }
        if (gpu_opt->n_dev_use == 0)
        {
            gmx_fatal(FARGS, "Empty GPU ID string encountered.\n%s\n",
                      invalid_gpuid_hint);
        }

        gpu_opt->bUserSet = TRUE;
    }
}

void gmx_hardware_info_free(gmx_hw_info_t *hwinfo)
{
    int ret;

    ret = tMPI_Thread_mutex_lock(&hw_info_lock);
    if (ret != 0)
    {
        gmx_fatal(FARGS, "Error locking hwinfo mutex: %s", strerror(errno));
    }

    /* decrease the reference counter */
    n_hwinfo--;


    if (hwinfo != hwinfo_g)
    {
        gmx_incons("hwinfo < hwinfo_g");
    }

    if (n_hwinfo < 0)
    {
        gmx_incons("n_hwinfo < 0");
    }

    if (n_hwinfo == 0)
    {
        delete hwinfo_g->cpuInfo;
        delete hwinfo_g->hardwareTopology;
        free_gpu_info(&hwinfo_g->gpu_info);
        sfree(hwinfo_g);
    }

    ret = tMPI_Thread_mutex_unlock(&hw_info_lock);
    if (ret != 0)
    {
        gmx_fatal(FARGS, "Error unlocking hwinfo mutex: %s", strerror(errno));
    }
}
