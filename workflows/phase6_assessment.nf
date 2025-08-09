#!/usr/bin/env nextflow

nextflow.enable.dsl=2

workflow PHASE6_ASSESSMENT {
    take:
        statistical_results
        benchmark_results
    
    main:
        // Assess ecosystem maturity
        ASSESS_ECOSYSTEM_MATURITY(benchmark_results)
        
        // Evaluate framework capabilities
        EVALUATE_FRAMEWORK_CAPABILITIES(benchmark_results)
        
        // Generate recommendations
        GENERATE_RECOMMENDATIONS(
            ASSESS_ECOSYSTEM_MATURITY.out.ecosystem_assessment,
            EVALUATE_FRAMEWORK_CAPABILITIES.out.framework_evaluation,
            statistical_results
        )
        
        // Generate final comprehensive report
        GENERATE_FINAL_REPORT(
            statistical_results,
            ASSESS_ECOSYSTEM_MATURITY.out.ecosystem_assessment,
            EVALUATE_FRAMEWORK_CAPABILITIES.out.framework_evaluation,
            GENERATE_RECOMMENDATIONS.out.recommendations
        )
    
    emit:
        final_report = GENERATE_FINAL_REPORT.out.final_report
        recommendations = GENERATE_RECOMMENDATIONS.out.recommendations
}

process ASSESS_ECOSYSTEM_MATURITY {
    tag "Ecosystem Maturity Assessment"
    label 'cpu_intensive'
    
    input:
    path benchmark_results
    
    output:
    path "ecosystem_assessment.json"
    
    script:
    """
    python /app/scripts/assess_ecosystem_maturity.py \\
        --benchmark-results ${benchmark_results} \\
        --output ecosystem_assessment.json
    """
}

process EVALUATE_FRAMEWORK_CAPABILITIES {
    tag "Framework Capabilities Evaluation"
    label 'cpu_intensive'
    
    input:
    path benchmark_results
    
    output:
    path "framework_evaluation.json"
    
    script:
    """
    python /app/scripts/evaluate_framework_capabilities.py \\
        --benchmark-results ${benchmark_results} \\
        --output framework_evaluation.json
    """
}

process GENERATE_RECOMMENDATIONS {
    tag "Recommendations Generation"
    label 'cpu_intensive'
    
    input:
    path ecosystem_assessment
    path framework_evaluation
    path statistical_results
    
    output:
    path "recommendations.json"
    
    script:
    """
    python /app/scripts/generate_recommendations.py \\
        --ecosystem-assessment ${ecosystem_assessment} \\
        --framework-evaluation ${framework_evaluation} \\
        --statistical-results ${statistical_results} \\
        --output recommendations.json
    """
}

process GENERATE_FINAL_REPORT {
    tag "Final Report Generation"
    label 'cpu_intensive'
    
    input:
    path statistical_results
    path ecosystem_assessment
    path framework_evaluation
    path recommendations
    
    output:
    path "final_comprehensive_report.md"
    
    script:
    """
    python /app/scripts/generate_final_report.py \\
        --statistical-results ${statistical_results} \\
        --ecosystem-assessment ${ecosystem_assessment} \\
        --framework-evaluation ${framework_evaluation} \\
        --recommendations ${recommendations} \\
        --output final_comprehensive_report.md
    """
} 