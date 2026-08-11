#!/bin/bash
# Quick Start Guide - Report Scripts

echo "=========================================="
echo "  Quick Start - Report Scripts"
echo "=========================================="
echo ""
echo "This quick guide shows how to use the report scripts."
echo ""

echo "1. ACTIVATE ENVIRONMENT"
echo "   conda activate vfnet"
echo ""

echo "2. AVAILABLE SCRIPTS"
echo ""
echo "   a) Classification Metrics (Tutorial 7.1)"
echo "      python scripts/reports/classification_metrics.py scripts/configs/reports/classification_metrics.yaml"
echo ""
echo "   b) Classification Times (Tutorial 7.2)"
echo "      python scripts/reports/classification_times.py scripts/configs/reports/classification_times.yaml"
echo ""
echo "   c) Accuracy by Curvatures (Tutorial 7.4)"
echo "      python scripts/reports/accuracy_by_curvatures.py scripts/configs/reports/accuracy_by_curvatures.yaml"
echo ""
echo "   d) Compare Models (Tutorial 7.31)"
echo "      python scripts/reports/compare_models.py scripts/configs/reports/compare_models.yaml"
echo ""

echo "3. INTERACTIVE MENU"
echo "   bash scripts/reports/run_reports.sh"
echo ""

echo "4. CREATE CUSTOM CONFIGURATION"
echo "   cp scripts/configs/reports/classification_metrics.yaml my_config.yaml"
echo "   vim my_config.yaml"
echo "   python scripts/reports/classification_metrics.py my_config.yaml"
echo ""

echo "5. DOCUMENTATION"
echo "   - Complete README:      scripts/reports/README.md"
echo "   - Examples:             scripts/configs/reports/EXAMPLES.md"
echo "   - Migration summary:    scripts/reports/MIGRATION_SUMMARY.md"
echo ""

echo "=========================================="
echo "For more information, see:"
echo "  cat scripts/reports/README.md"
echo "=========================================="
