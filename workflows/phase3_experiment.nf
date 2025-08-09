#!/usr/bin/env nextflow

nextflow.enable.dsl=2

workflow PHASE3_EXPERIMENT {
    take:
        implementations
    
    main:
        // Setup experimental environment
        SETUP_EXPERIMENTAL_ENVIRONMENT(implementations)
        
        // Run pilot studies
        RUN_PILOT_STUDIES(SETUP_EXPERIMENTAL_ENVIRONMENT.out.environment_config)
        
        // Validate pilot results
        VALIDATE_PILOT_RESULTS(RUN_PILOT_STUDIES.out.pilot_results)
        
        // Generate pilot report
        GENERATE_PILOT_REPORT(VALIDATE_PILOT_RESULTS.out.validated_pilot_results)
    
    emit:
        validated_implementations = VALIDATE_PILOT_RESULTS.out.validated_implementations
        environment_config = SETUP_EXPERIMENTAL_ENVIRONMENT.out.environment_config
        pilot_report = GENERATE_PILOT_REPORT.out.report
}

process SETUP_EXPERIMENTAL_ENVIRONMENT {
    tag "Experimental Environment Setup"
    label 'cpu_intensive'
    
    input:
    path implementations
    
    output:
    path "environment_config.json"
    
    script:
    """
    python /app/scripts/setup_experimental_environment.py \\
        --implementations ${implementations} \\
        --output environment_config.json
    """
}

process RUN_PILOT_STUDIES {
    tag "Pilot Studies Execution"
    label 'cpu_intensive'
    
    input:
    path environment_config
    
    output:
    path "pilot_results.json"
    
    script:
    """
    python /app/scripts/run_pilot_studies.py \\
        --config ${environment_config} \\
        --output pilot_results.json
    """
}

process VALIDATE_PILOT_RESULTS {
    tag "Pilot Results Validation"
    label 'cpu_intensive'
    
    input:
    path pilot_results
    
    output:
    path "validated_pilot_results.json"
    path "validated_implementations.json"
    
    script:
    """
    python /app/scripts/validate_pilot_results.py \\
        --results ${pilot_results} \\
        --output-validated validated_pilot_results.json \\
        --output-implementations validated_implementations.json
    """
}

process GENERATE_PILOT_REPORT {
    tag "Pilot Report Generation"
    label 'cpu_intensive'
    
    input:
    path validated_pilot_results
    
    output:
    path "pilot_report.md"
    
    script:
    """
    python /app/scripts/generate_pilot_report.py \\
        --results ${validated_pilot_results} \\
        --output pilot_report.md
    """
} 