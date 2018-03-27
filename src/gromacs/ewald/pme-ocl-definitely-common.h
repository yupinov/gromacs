#ifndef PMEOCLDEFINITELYCOMMON_H
#define PMEOCLDEFINITELYCOMMON_H

/*
inline int actualExecutionWidth()
{
    return 32;
}
*/

//#define warp_size
//actualExecutionWidth()


#define c_solveMaxWarpsPerBlock 8
//! Solving kernel max block size in threads
#define c_solveMaxThreadsPerBlock (c_solveMaxWarpsPerBlock * warp_size)



#define c_virialAndEnergyCount 7

#endif // PMEOCLDEFINITELYCOMMON_H
