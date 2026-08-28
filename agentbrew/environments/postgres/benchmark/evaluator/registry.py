"""Custom evaluator functions for MCPMark Postgres tasks."""
# pylint: disable=line-too-long,import-outside-toplevel

from __future__ import annotations

from typing import Callable, Tuple

from agentbrew.environments.postgres.benchmark.evaluator.utils import (
    VerificationStep,
    expected_customers_args,
    run_legacy_verify,
    run_step_based_verify,
)

Verifier = Callable[..., object]
COMPARISON_FUNCTIONS: dict[str, Verifier] = {}


def compare_func(*, name: str) -> Callable[[Verifier], Verifier]:
    """Register a verifier by the operator name stored in benchmark task JSON."""

    def decorator(function: Verifier) -> Verifier:
        COMPARISON_FUNCTIONS[name] = function
        return function

    return decorator


def _run_steps(module_path: str, task_name: str, steps: list[VerificationStep]) -> Tuple[bool, str]:
    return run_step_based_verify(module_path=module_path, task_name=task_name, steps=steps)


def _run_legacy(module_path: str, task_name: str) -> Tuple[bool, str]:
    return run_legacy_verify(module_path=module_path, task_name=task_name)


@compare_func(name="postgres_customer_data_migration_verifier")
async def postgres_customer_data_migration_verifier(_x: dict, *_args, **_kwargs) -> Tuple[bool, str]:
    return _run_steps(
        "agentbrew.environments.postgres.benchmark.evaluator.chinook.customer_data_migration.verify",
        "Customer Data Migration",
        [VerificationStep("Migrated customers match expected dataset", "verify_migrated_customers", expected_customers_args)],
    )


@compare_func(name="postgres_employee_hierarchy_verifier")
async def postgres_employee_hierarchy_verifier(_x: dict, *_args, **_kwargs) -> Tuple[bool, str]:
    return _run_steps(
        "agentbrew.environments.postgres.benchmark.evaluator.chinook.employee_hierarchy_management.verify",
        "Employee Hierarchy Management",
        [
            VerificationStep("Employee count and titles", "verify_employee_count_and_titles"),
            VerificationStep("Specific employee records", "verify_specific_employees"),
            VerificationStep("Customer assignments", "verify_customer_assignments"),
            VerificationStep("Performance table", "verify_performance_table"),
            VerificationStep("Deletion and promotion", "verify_employee_deletion_and_promotion"),
            VerificationStep("Salary column", "verify_salary_column"),
        ],
    )


@compare_func(name="postgres_sales_music_charts_verifier")
async def postgres_sales_music_charts_verifier(_x: dict, *_args, **_kwargs) -> Tuple[bool, str]:
    return _run_steps(
        "agentbrew.environments.postgres.benchmark.evaluator.chinook.sales_and_music_charts.verify",
        "Sales and Music Charts",
        [
            VerificationStep("Monthly sales summary", "verify_monthly_sales_results"),
            VerificationStep("Music charts results", "verify_music_charts_results"),
        ],
    )


@compare_func(name="postgres_customer_analysis_fix_verifier")
async def postgres_customer_analysis_fix_verifier(_x: dict, *_args, **_kwargs) -> Tuple[bool, str]:
    return _run_steps(
        "agentbrew.environments.postgres.benchmark.evaluator.dvdrental.customer_analysis_fix.verify",
        "Customer Analysis Fix",
        [VerificationStep("customer_analysis_fixed table matches expectation", "verify_customer_analysis_fixed_table")],
    )


@compare_func(name="postgres_customer_analytics_optimization_verifier")
async def postgres_customer_analytics_optimization_verifier(_x: dict, *_args, **_kwargs) -> Tuple[bool, str]:
    return _run_legacy(
        "agentbrew.environments.postgres.benchmark.evaluator.dvdrental.customer_analytics_optimization.verify",
        "Customer Analytics Optimization",
    )


@compare_func(name="postgres_film_inventory_verifier")
async def postgres_film_inventory_verifier(_x: dict, *_args, **_kwargs) -> Tuple[bool, str]:
    return _run_steps(
        "agentbrew.environments.postgres.benchmark.evaluator.dvdrental.film_inventory_management.verify",
        "Film Inventory Management",
        [
            VerificationStep("New films inserted", "check_new_films"),
            VerificationStep("Inventory records inserted", "check_inventory_records"),
            VerificationStep("available_films table", "check_available_films_table"),
            VerificationStep("Inventory cleanup", "check_inventory_cleanup"),
            VerificationStep("film_inventory_summary table", "check_summary_table"),
        ],
    )


@compare_func(name="postgres_employee_demographics_verifier")
async def postgres_employee_demographics_verifier(_x: dict, *_args, **_kwargs) -> Tuple[bool, str]:
    return _run_steps(
        "agentbrew.environments.postgres.benchmark.evaluator.employees.employee_demographics_report.verify",
        "Employee Demographics Report",
        [
            VerificationStep("Gender statistics", "verify_gender_statistics_results"),
            VerificationStep("Age group analysis", "verify_age_group_results"),
            VerificationStep("Birth month distribution", "verify_birth_month_results"),
            VerificationStep("Hiring year summary", "verify_hiring_year_results"),
        ],
    )


@compare_func(name="postgres_employee_performance_verifier")
async def postgres_employee_performance_verifier(_x: dict, *_args, **_kwargs) -> Tuple[bool, str]:
    return _run_steps(
        "agentbrew.environments.postgres.benchmark.evaluator.employees.employee_performance_analysis.verify",
        "Employee Performance Analysis",
        [
            VerificationStep("Performance analysis table", "verify_performance_results"),
            VerificationStep("Department salary analysis", "verify_department_results"),
        ],
    )


@compare_func(name="postgres_employee_project_tracking_verifier")
async def postgres_employee_project_tracking_verifier(_x: dict, *_args, **_kwargs) -> Tuple[bool, str]:
    return _run_steps(
        "agentbrew.environments.postgres.benchmark.evaluator.employees.employee_project_tracking.verify",
        "Employee Project Tracking",
        [
            VerificationStep("Table structures", "verify_table_structures"),
            VerificationStep("Indexes", "verify_indexes"),
            VerificationStep("Project data", "verify_project_data"),
            VerificationStep("Assignment data", "verify_assignment_data"),
            VerificationStep("Milestone data", "verify_milestone_data"),
        ],
    )


@compare_func(name="postgres_employee_retention_verifier")
async def postgres_employee_retention_verifier(_x: dict, *_args, **_kwargs) -> Tuple[bool, str]:
    return _run_steps(
        "agentbrew.environments.postgres.benchmark.evaluator.employees.employee_retention_analysis.verify",
        "Employee Retention Analysis",
        [
            VerificationStep("Retention analysis", "verify_retention_analysis_results"),
            VerificationStep("High risk employees", "verify_high_risk_results"),
            VerificationStep("Turnover trends", "verify_turnover_trend_results"),
        ],
    )


@compare_func(name="postgres_executive_dashboard_automation_verifier")
async def postgres_executive_dashboard_automation_verifier(_x: dict, *_args, **_kwargs) -> Tuple[bool, str]:
    return _run_steps(
        "agentbrew.environments.postgres.benchmark.evaluator.employees.executive_dashboard_automation.verify",
        "Executive Dashboard Automation",
        [
            VerificationStep("Materialized views", "verify_materialized_views"),
            VerificationStep("Stored procedures", "verify_stored_procedures"),
            VerificationStep("Triggers", "verify_triggers"),
            VerificationStep("Procedure execution", "verify_procedure_execution"),
            VerificationStep("Indexes", "verify_indexes"),
        ],
    )


@compare_func(name="postgres_management_structure_analysis_verifier")
async def postgres_management_structure_analysis_verifier(_x: dict, *_args, **_kwargs) -> Tuple[bool, str]:
    return _run_steps(
        "agentbrew.environments.postgres.benchmark.evaluator.employees.management_structure_analysis.verify",
        "Management Structure Analysis",
        [
            VerificationStep("Manager profile", "verify_manager_profile_results"),
            VerificationStep("Department leadership", "verify_department_leadership_results"),
            VerificationStep("Management transitions", "verify_management_transitions_results"),
            VerificationStep("Span of control", "verify_span_of_control_results"),
        ],
    )


@compare_func(name="postgres_consistency_enforcement_verifier")
async def postgres_consistency_enforcement_verifier(_x: dict, *_args, **_kwargs) -> Tuple[bool, str]:
    return _run_steps(
        "agentbrew.environments.postgres.benchmark.evaluator.lego.consistency_enforcement.verify",
        "Consistency Enforcement",
        [
            VerificationStep("Data consistency", "verify_data_consistency"),
            VerificationStep("Constraint triggers exist", "verify_constraint_triggers_exist"),
            VerificationStep("Violations are blocked", "verify_violation_is_blocked"),
            VerificationStep("Deferred transaction allowed", "verify_deferred_transaction_is_allowed"),
        ],
    )


@compare_func(name="postgres_database_security_policies_verifier")
async def postgres_database_security_policies_verifier(_x: dict, *_args, **_kwargs) -> Tuple[bool, str]:
    return _run_steps(
        "agentbrew.environments.postgres.benchmark.evaluator.lego.database_security_policies.verify",
        "Database Security Policies",
        [
            VerificationStep("Role creation", "verify_role_creation"),
            VerificationStep("RLS enabled", "verify_rls_enabled"),
            VerificationStep("RLS policies", "verify_rls_policies"),
            VerificationStep("Theme function", "verify_theme_function"),
            VerificationStep("Theme analyst access", "test_theme_analyst_access"),
        ],
    )


@compare_func(name="postgres_transactional_inventory_transfer_verifier")
async def postgres_transactional_inventory_transfer_verifier(_x: dict, *_args, **_kwargs) -> Tuple[bool, str]:
    return _run_steps(
        "agentbrew.environments.postgres.benchmark.evaluator.lego.transactional_inventory_transfer.verify",
        "Transactional Inventory Transfer",
        [
            VerificationStep("System components", "verify_system_components"),
            VerificationStep("Successful transfer with audit", "verify_successful_transfer_with_audit"),
            VerificationStep("New part transfer", "verify_new_part_transfer"),
            VerificationStep("Business rule validation", "verify_business_rule_validation"),
            VerificationStep("Insufficient quantity error", "verify_insufficient_quantity_error"),
            VerificationStep("Invalid inventory error", "verify_invalid_inventory_error"),
            VerificationStep("Audit logging", "verify_audit_logging"),
            VerificationStep("Exact quantity transfer", "verify_exact_quantity_transfer"),
        ],
    )


@compare_func(name="postgres_rls_business_access_verifier")
async def postgres_rls_business_access_verifier(_x: dict, *_args, **_kwargs) -> Tuple[bool, str]:
    return _run_legacy(
        "agentbrew.environments.postgres.benchmark.evaluator.security.rls_business_access.verify",
        "RLS Business Access",
    )


@compare_func(name="postgres_security_verifier")
async def postgres_security_verifier(_x: dict, *_args, **_kwargs) -> Tuple[bool, str]:
    return _run_legacy(
        "agentbrew.environments.postgres.benchmark.evaluator.security.user_permission_audit.verify",
        "User Permission Audit",
    )


@compare_func(name="postgres_baseball_player_analysis_verifier")
async def postgres_baseball_player_analysis_verifier(_x: dict, *_args, **_kwargs) -> Tuple[bool, str]:
    return _run_steps(
        "agentbrew.environments.postgres.benchmark.evaluator.sports.baseball_player_analysis.verify",
        "Baseball Player Analysis",
        [VerificationStep("Baseball player analysis table", "verify_baseball_player_analysis_table")],
    )


@compare_func(name="postgres_participant_report_optimization_verifier")
async def postgres_participant_report_optimization_verifier(_x: dict, *_args, **_kwargs) -> Tuple[bool, str]:
    return _run_steps(
        "agentbrew.environments.postgres.benchmark.evaluator.sports.participant_report_optimization.verify",
        "Participant Report Optimization",
        [
            VerificationStep("Participant report data", "verify_report_data"),
            VerificationStep("Performance optimization", "verify_performance_optimization"),
        ],
    )


@compare_func(name="postgres_team_roster_management_verifier")
async def postgres_team_roster_management_verifier(_x: dict, *_args, **_kwargs) -> Tuple[bool, str]:
    return _run_steps(
        "agentbrew.environments.postgres.benchmark.evaluator.sports.team_roster_management.verify",
        "Team Roster Management",
        [
            VerificationStep("Player evaluation table", "verify_player_evaluation_table"),
            VerificationStep("Injury status table", "verify_injury_status_table"),
            VerificationStep("Team performance summary", "verify_summary_table"),
        ],
    )


@compare_func(name="postgres_dba_vector_analysis_verifier")
async def postgres_dba_vector_analysis_verifier(_x: dict, *_args, **_kwargs) -> Tuple[bool, str]:
    return _run_steps(
        "agentbrew.environments.postgres.benchmark.evaluator.vectors.dba_vector_analysis.verify",
        "DBA Vector Analysis",
        [
            VerificationStep("Vector analysis columns", "verify_vector_analysis_columns"),
            VerificationStep("Vector storage consumption", "verify_vector_analysis_storage_consumption"),
            VerificationStep("Vector indices", "verify_vector_analysis_indices"),
            VerificationStep("No extra analysis tables", "verify_no_extra_analysis_tables"),
        ],
    )
