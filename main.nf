#!/usr/bin/env nextflow

nextflow.enable.dsl=2

// Import workflow modules
include { PHASE1_SELECTION } from './workflows/phase1_selection.nf'
include { PHASE2_IMPLEMENTATION } from './workflows/phase2_implementation.nf'
include { PHASE3_EXPERIMENT } from './workflows/phase3_experiment.nf'
include { PHASE4_BENCHMARK } from './workflows/phase4_benchmark.nf'
include { PHASE5_ANALYSIS } from './workflows/phase5_analysis.nf'
include { PHASE6_ASSESSMENT } from './workflows/phase6_assessment.nf'

workflow {
    // Phase 1: Framework Selection and Validation
    PHASE1_SELECTION()
    
    // Phase 2: Implementation and Validation
    PHASE2_IMPLEMENTATION(PHASE1_SELECTION.out.selected_frameworks)
    
    // Phase 3: Experimental Setup and Pilot Studies
    PHASE3_EXPERIMENT(PHASE2_IMPLEMENTATION.out.implementations)
    
    // Phase 4: Benchmark Execution
    PHASE4_BENCHMARK(
        PHASE3_EXPERIMENT.out.validated_implementations,
        PHASE3_EXPERIMENT.out.environment_config
    )
    
    // Phase 5: Statistical Analysis
    PHASE5_ANALYSIS(PHASE4_BENCHMARK.out.benchmark_results)
    
    // Phase 6: Ecosystem Assessment and Final Report
    PHASE6_ASSESSMENT(
        PHASE5_ANALYSIS.out.statistical_results,
        PHASE4_BENCHMARK.out.benchmark_results
    )
} 