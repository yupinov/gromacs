/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 2016,2017, by the GROMACS development team, led by
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

#include "hardwareassign.h"

#include "config.h"

#include <cstring>

#include <algorithm>
#include <string>

#include "thread_mpi/threads.h"

#include "gromacs/gmxlib/network.h"
#include "gromacs/gpu_utils/gpu_utils.h"
#include "gromacs/hardware/detecthardware.h"
#include "gromacs/hardware/gpu_hw_info.h"
#include "gromacs/mdtypes/commrec.h"
#include "gromacs/utility/basenetwork.h"
#include "gromacs/utility/exceptions.h"
#include "gromacs/utility/fatalerror.h"
#include "gromacs/utility/gmxassert.h"
#include "gromacs/utility/logger.h"
#include "gromacs/utility/programcontext.h"
#include "gromacs/utility/smalloc.h"
#include "gromacs/utility/stringutil.h"
#include "gromacs/utility/sysinfo.h"

#define HOSTNAMELEN 80

/*! \internal \brief
 * Prints GPU information strings on this node into the stderr and log.
 * Only used for logging errors in heterogenous MPI configurations.
 */
static void print_gpu_detection_stats(const gmx::MDLogger  &mdlog,
                                      const gmx_gpu_info_t *gpu_info)
{
    char onhost[HOSTNAMELEN+10];
    int  ngpu;

    if (!gpu_info->bDetectGPUs)
    {
        /* We skipped the detection, so don't print detection stats */
        return;
    }

    ngpu = gpu_info->n_dev;

    /* We only print the detection on one, of possibly multiple, nodes */
    std::strncpy(onhost, " on host ", 10);
    gmx_gethostname(onhost + 9, HOSTNAMELEN);

    if (ngpu > 0)
    {
        std::string gpuDesc = sprint_gpus(gpu_info);
        GMX_LOG(mdlog.warning).asParagraph().appendTextFormatted(
                "%d GPU%s detected%s:\n%s",
                ngpu, (ngpu > 1) ? "s" : "", onhost, gpuDesc.c_str());
    }
    else
    {
        GMX_LOG(mdlog.warning).asParagraph().appendTextFormatted("No GPUs detected%s", onhost);
    }
    // FIXME: This currently only logs on the master rank, which defeats the purpose.
    // A new MDLogger option is required for printing to stderr on all ranks.
    // There is also a question of MPI reduction of the outputs, see Redmine issue #1505.
}

/*! \brief
 * This communicates the maximum possible GPU task counts within every node.
 *
 * On each rank we want to have the same node-local gpu_opt->dev_use array of assignments of GPU tasks to GPUs.
 * This function communicates the desired number of GPU tasks on this rank
 * with the rest of ranks of the node, and uses prefix sums to get a rank-local index
 * into gpu_opt->dev_use (devUseIndex).
 *
 * \param[in]  cr                    Communication structure
 * \param[in]  gpuTasksCountRank     Number of GPU tasks on this rank
 * \param[out] devUseIndex           Index into gpu_opt->dev_use
 * \param[out] devUseCountNode       Total number of GPU tasks on a physical node
 */
static void discoverGpuTasksCountsNode(const t_commrec *cr, int gpuTasksCountRank,
                                       int *devUseIndex, int *devUseCountNode)
{
    GMX_ASSERT(devUseIndex, "devUseIndex pointer is required");
    GMX_ASSERT(devUseCountNode, "devUseCountNode pointer is required");
    // Default values which are valid for a serial case
    // Index into gpu_opt->dev_use
    *devUseIndex      = 0;
    // Initial value for a prefix sum
    *devUseCountNode = gpuTasksCountRank;
    if ((PAR(cr) || MULTISIM(cr))) // Checking for MPI to be initialized
    {
#if GMX_MPI
        /* Prefix sums of the per-rank GPU task counts */
        MPI_Scan(const_cast<int *>(&gpuTasksCountRank), devUseIndex, 1, MPI_INT, MPI_SUM, cr->mpi_comm_physicalnode);
        /* Getting total amount of GPU tasks on this node - the last prefix sum is full sum */
        *devUseCountNode = *devUseIndex;
        const int lastRank = cr->nrank_intranode - 1;
        MPI_Bcast(devUseCountNode, 1, MPI_INT, lastRank, cr->mpi_comm_physicalnode);
        /* MPI_Scan is inclusive prefix sum, we need exclusive for the starting indices */
        *devUseIndex -= gpuTasksCountRank;
#endif
    }
}

/*! \brief Return whether all selected GPUs are compatible.
 *
 * Given the list of selected GPU device IDs in \c gpu_opt and
 * detected GPUs in \c gpu_info, return whether all selected GPUs are
 * compatible. If not, place a suitable string in \c errorMessage.
 *
 * \param[in]   gpu_info      pointer to structure holding GPU information
 * \param[in]   gpu_opt       pointer to structure holding GPU options
 * \param[out]  errorMessage  pointer to string to hold a possible error message (is not updated when returning true)
 * \returns                   true if every requested GPU is compatible
 */
static bool checkGpuSelection(const gmx_gpu_info_t *gpu_info,
                              const gmx_gpu_opt_t  *gpu_opt,
                              std::string          *errorMessage)
{
    GMX_ASSERT(gpu_info, "Invalid gpu_info");

    bool        allOK   = true;
    std::string message = "Some of the requested GPUs do not exist, behave strangely, or are not compatible:\n";
    for (int i = 0; i < gpu_opt->n_dev_use; i++)
    {
        GMX_ASSERT(gpu_opt, "Invalid gpu_opt");
        GMX_ASSERT(gpu_opt->dev_use, "Invalid gpu_opt->dev_use");

        int id     = gpu_opt->dev_use[i];
        if (!isGpuCompatible(gpu_info, id))
        {
            allOK    = false;
            message += gmx::formatString("    GPU #%d: %s\n",
                                         id,
                                         getGpuCompatibilityDescription(gpu_info, id));
        }
    }
    if (!allOK && errorMessage)
    {
        *errorMessage = message;
    }
    return allOK;
}

/*! \brief Select the compatible GPUs
 *
 * This function filters gpu_info->gpu_dev for compatible gpus based
 * on the previously run compatibility tests. Sets
 * gpu_info->dev_compatible and gpu_info->n_dev_compatible.
 *
 * \param[in]     gpu_info    pointer to structure holding GPU information
 * \param[out]    gpu_opt     pointer to structure holding GPU options
 */
static void pickCompatibleGpus(const gmx_gpu_info_t *gpu_info,
                               gmx_gpu_opt_t        *gpu_opt)
{
    GMX_ASSERT(gpu_info, "Invalid gpu_info");
    GMX_ASSERT(gpu_opt, "Invalid gpu_opt");

    // Possible minor over-allocation here, but not important for anything
    gpu_opt->n_dev_compatible = 0;
    snew(gpu_opt->dev_compatible, gpu_info->n_dev);
    for (int i = 0; i < gpu_info->n_dev; i++)
    {
        GMX_ASSERT(gpu_info->gpu_dev, "Invalid gpu_info->gpu_dev");
        if (isGpuCompatible(gpu_info, i))
        {
            gpu_opt->dev_compatible[gpu_opt->n_dev_compatible] = i;
            gpu_opt->n_dev_compatible++;
        }
    }
}

void GpuTaskAssignmentManager::assignRankGpuIds()
{
    GMX_RELEASE_ASSERT(devUseCountNode_ >= 1,
                       gmx::formatString("Invalid limit (%d) for the number of GPUs (detected %d compatible GPUs)",
                                         devUseCountNode_, gpuOpt_->n_dev_compatible).c_str());

    if (gpuOpt_->n_dev_compatible == 0)
    {
        char host[HOSTNAMELEN];

        gmx_gethostname(host, HOSTNAMELEN);
        gmx_fatal(FARGS, "A GPU was requested on host %s, but no compatible GPUs were detected. If you intended to use GPU acceleration in a parallel run, you can either avoid using the nodes that don't have GPUs or place CPU tasks on these nodes.", host);
    }

    int nshare = 1; /* Max. number of GPU tasks sharing a single GPU */
    if (devUseCountNode_ > gpuOpt_->n_dev_compatible)
    {
        if (devUseCountNode_ % gpuOpt_->n_dev_compatible == 0)
        {
            nshare = gmx_gpu_sharing_supported() ? (devUseCountNode_ / gpuOpt_->n_dev_compatible) : 1;
        }
        else
        {
            const bool firstGpuRank = (devUseIndex_ == 0) && !tasksToAssign_.empty();
            if (firstGpuRank) // Printing the error message only on one rank
            {
                gmx_fatal(FARGS, "The number of GPU tasks (%d) is not a multiple of the actual number of GPUs (%d). Select a different number of MPI ranks or use the -gpu_id option to manually specify the GPUs to be used.",
                          devUseCountNode_, gpuOpt_->n_dev_compatible);
            }

#if GMX_MPI
            /* We use a global barrier to prevent ranks from continuing with
             * an invalid setup.
             */
            MPI_Barrier(MPI_COMM_WORLD);
#endif
        }
    }

    /* Here we will waste GPUs when dev_use_count_node < gpu_opt->n_dev_compatible */
    gpuOpt_->n_dev_use = std::min(gpuOpt_->n_dev_compatible * nshare, devUseCountNode_);
    if (!gmx_multiple_gpu_per_node_supported())
    {
        gpuOpt_->n_dev_use = std::min(gpuOpt_->n_dev_use, 1);
    }
    snew(gpuOpt_->dev_use, gpuOpt_->n_dev_use);
    for (int i = 0; i != gpuOpt_->n_dev_use; ++i)
    {
        /* TODO: improve this implementation: either sort GPUs or remove the weakest here */
        gpuOpt_->dev_use[i] = gpuOpt_->dev_compatible[i / nshare];
    }
}

GpuContext GpuTaskAssignmentManager::getGpuContext(int gpuIndex)
{
    if (gpuIndex < 0 || gpuIndex >= gpuOpt_->n_dev_use)
    {
        std::string errorString = gmx::formatString("Trying to assign a non-existent GPU: "
                                                    "there are %d %s-selected GPU(s), but #%d was requested.",
                                                    gpuOpt_->n_dev_use, gpuOpt_->bUserSet ? "user" : "auto", gpuIndex);
        gmx_incons(errorString.c_str());
    }

    const int  gpuId = get_gpu_device_id(gpuInfo_, gpuOpt_, gpuIndex);
    GpuContext context;
    context.gpuId_ = gpuId;
    // currently GPU ID is the same as the index into gpu_dev
    context.gpuInfo_ = reinterpret_cast<gmx_device_info_t *>(reinterpret_cast<char *>(gpuInfo_->gpu_dev)
                                                             + gpuId * sizeof_gpu_dev_info());
    return context;
}

void GpuTaskAssignmentManager::registerGpuTask(GpuTask task)
{
    /* Bail if binary is not compiled with GPU acceleration */
    if (GMX_GPU == GMX_GPU_NONE)
    {
        gmx_fatal(FARGS, "GPU acceleration requested, but %s was compiled without GPU support!",
                  gmx::getProgramContext().displayName());
    }

    tasksToAssign_.insert(task);
}

void GpuTaskAssignmentManager::selectRankGpus(const gmx::MDLogger &mdlog, const t_commrec *cr)
{
    /* This should be called after all the registerGpuTask() calls,
     * so we know how many GPUs this process can use at most.
     * The actual used GPU count at any point can potentially be smaller.
     */
    discoverGpuTasksCountsNode(cr, tasksToAssign_.size(), &devUseIndex_, &devUseCountNode_);

    if (devUseCountNode_ == 0)
    {
        /* Ignore (potentially) manually selected GPUs */
        gpuOpt_->n_dev_use = 0;
        return;
    }

    if (gpuOpt_->bUserSet)
    {
        /* Check the GPU IDs passed by the user.
         * (GPU IDs have been parsed by gmx_parse_gpu_ids before)
         */
        std::string errorMessage;
        if (!checkGpuSelection(gpuInfo_, gpuOpt_, &errorMessage))
        {
            const bool canHaveHeterogeneousNodes = GMX_LIB_MPI && PAR(cr);
            if (canHaveHeterogeneousNodes)
            {
                print_gpu_detection_stats(mdlog, gpuInfo_);
            }
            gmx_fatal(FARGS, errorMessage.c_str());
        }

        /* Check whether the number of manually provided GPU IDs corresponds to the total number of GPU tasks */
        if (gpuOpt_->n_dev_use != devUseCountNode_)
        {
            /* TODO: this was moved from gmx_check_hw_runconf_consistency() and uses same strings */
            const std::string pernode            = GMX_LIB_MPI ? " per node" : "";
            const std::string gpuTasksUserPlural = (gpuOpt_->n_dev_use > 1) ? "s" : "";
            const char       *programName        = gmx::getProgramContext().displayName();
            const int         gpuTasksNodeCount  = devUseCountNode_;
            const std::string gpuTasksNodePlural = (gpuTasksNodeCount > 1) ? "s" : "";
            gmx_fatal(FARGS,
                      "Incorrect launch configuration: mismatching number of GPU tasks and GPUs%s.\n"
                      "%s was started with %d GPU task%s%s in total, but you've provided %d GPU ID%s.",
                      pernode.c_str(), programName, gpuTasksNodeCount, gpuTasksNodePlural.c_str(), pernode.c_str(),
                      gpuOpt_->n_dev_use, gpuTasksUserPlural.c_str());
        }
    }
    else if (getenv("GMX_EMULATE_GPU") == nullptr)
    {
        pickCompatibleGpus(gpuInfo_, gpuOpt_);
        /* Assign GPUs to ranks automatically. Intra-rank GPU to task assignment happens in selectTasksGpus. */
        assignRankGpuIds();
    }
}

GpuContextsMap GpuTaskAssignmentManager::selectTasksGpus()
{
    /* Here we assign indices into gpu_opt->dev_use to the GPU tasks of this rank.
     * gpu_opt->dev_use had already been filled in selectRankGpus.
     */
    GpuContextsMap gpuContextsByTask;
    int            gpuIndex = devUseIndex_;
    if (tasksToAssign_.count(GpuTask::NB) > 0)
    {
        gpuContextsByTask[GpuTask::NB] = getGpuContext(gpuIndex);
    }
    return gpuContextsByTask;
}

gmx_device_info_t *GpuTaskManager::gpuInfo(GpuTask task) const
{
    gmx_device_info_t *gpuInfo = nullptr;
    try
    {
        gpuInfo = gpuContextsByTasks_.at(task).gpuInfo_;
    }
    catch (...)
    {
    }
    return gpuInfo;
}

int GpuTaskManager::gpuId(GpuTask task) const
{
    return gpuContextsByTasks_.at(task).gpuId_;
}

size_t GpuTaskManager::rankGpuTasksCount() const
{
    return gpuContextsByTasks_.size();
}

GpuTaskManager createGpuAssignment(const gmx::MDLogger &mdlog, const t_commrec *cr,
                                   const gmx_gpu_info_t &gpuInfo, gmx_gpu_opt_t &gpuOpt,
                                   bool useGpuNB)
{
    GpuTaskAssignmentManager assigner(&gpuInfo, &gpuOpt);
    if (useGpuNB && (cr->duty & DUTY_PP))
    {
        assigner.registerGpuTask(GpuTask::NB);
    }
    /* This chooses node-local GPU IDs */
    assigner.selectRankGpus(mdlog, cr);
    /* This sorts out the rank-local GPU to task assignment */
    auto gpuTasks = assigner.selectTasksGpus();
    return gpuTasks;
}
