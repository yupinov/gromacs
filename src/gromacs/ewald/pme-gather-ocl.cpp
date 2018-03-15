void pme_gpu_gather(PmeGpu                *pmeGpu,
                    PmeForceOutputHandling forceTreatment,
                    const float           *h_grid
                    )
{
    /* Copying the input CPU forces for reduction */
    if (forceTreatment != PmeForceOutputHandling::Set)
    {
        pme_gpu_copy_input_forces(pmeGpu);
    }

    cudaStream_t stream          = pmeGpu->archSpecific->pmeStream;
    const int    order           = pmeGpu->common->pme_order;
    const auto  *kernelParamsPtr = pmeGpu->kernelParams.get();

    if (!pme_gpu_performs_FFT(pmeGpu) || pme_gpu_is_testing(pmeGpu))
    {
        pme_gpu_copy_input_gather_grid(pmeGpu, const_cast<float *>(h_grid));
    }

    if (pme_gpu_is_testing(pmeGpu))
    {
        pme_gpu_copy_input_gather_atom_data(pmeGpu);
    }

    const int atomsPerBlock  =  (c_gatherMaxThreadsPerBlock / PME_SPREADGATHER_THREADS_PER_ATOM);
    GMX_ASSERT(!c_usePadding || !(PME_ATOM_DATA_ALIGNMENT % atomsPerBlock), "inconsistent atom data padding vs. gathering block size");

    dim3 nBlocks(pmeGpu->nAtomsPadded / atomsPerBlock);
    dim3 dimBlock(order, order, atomsPerBlock);

    const bool wrapX = true;
    const bool wrapY = true;
    GMX_UNUSED_VALUE(wrapX);
    GMX_UNUSED_VALUE(wrapY);

    // TODO test different cache configs

    pme_gpu_start_timing(pmeGpu, gtPME_GATHER);
    if (order == 4)
    {
        if (forceTreatment == PmeForceOutputHandling::Set)
        {
            pme_gather_kernel<4, true, wrapX, wrapY> <<< nBlocks, dimBlock, 0, stream>>> (*kernelParamsPtr);
        }
        else
        {
            pme_gather_kernel<4, false, wrapX, wrapY> <<< nBlocks, dimBlock, 0, stream>>> (*kernelParamsPtr);
        }
    }
    else
    {
        GMX_THROW(gmx::NotImplementedError("The code for pme_order != 4 is not implemented"));
    }
    CU_LAUNCH_ERR("pme_gather_kernel");
    pme_gpu_stop_timing(pmeGpu, gtPME_GATHER);

    pme_gpu_copy_output_forces(pmeGpu);
}
