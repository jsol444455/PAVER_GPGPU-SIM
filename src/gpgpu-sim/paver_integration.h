// =============================================================================
// paver_integration.h - PAVER Integration for GPGPU-Sim
// =============================================================================
//
// This file provides the interface between PAVER's graph partitioning output
// and the GPGPU-Sim runtime scheduler.
//
// Add this file to: src/gpgpu-sim/paver_integration.h
// =============================================================================

#ifndef PAVER_INTEGRATION_H
#define PAVER_INTEGRATION_H

#include "/home/m/gpgpu-sim/src/abstract_hardware_model.h"
#include <vector>
#include <map>
#include <string>

// =============================================================================
// PAVER Integration Functions
// =============================================================================

/**
 * Initialize PAVER scheduling for a kernel.
 * 
 * This function should be called after the kernel is created but before
 * any TBs are issued. It sets up the TB partitions and SM assignments.
 * 
 * @param kernel      The kernel to enable PAVER for
 * @param num_sms     Number of SMs in the GPU configuration
 */
void paver_setup_kernel(kernel_info_t* kernel, unsigned num_sms);

/**
 * Load PAVER configuration from a file.
 * 
 * File format (one line per partition):
 *   PARTITION_ID: TB_ID, TB_ID, TB_ID, ...
 * 
 * @param filename    Path to PAVER schedule file
 * @param partitions  Output: list of TB partitions
 * @param sm_assign   Output: SM to partition assignments
 * @return true on success
 */
bool paver_load_config(
    const std::string& filename,
    std::vector<std::vector<unsigned>>& partitions,
    std::map<unsigned, std::vector<unsigned>>& sm_assign
);

/**
 * Check if PAVER should be enabled for a kernel.
 * 
 * PAVER is beneficial when:
 * - Kernel has inter-TB data locality
 * - Kernel has enough TBs to benefit from partitioning
 * 
 * @param kernel  The kernel to check
 * @return true if PAVER should be used
 */
bool paver_should_enable(kernel_info_t* kernel);

#endif // PAVER_INTEGRATION_H
