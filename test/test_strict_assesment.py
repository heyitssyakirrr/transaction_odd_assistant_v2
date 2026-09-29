from __future__ import annotations

import asyncio
import json
import unittest
from pathlib import Path

from app.adapters.llm_client import LlmServiceError, OpenAICompatibleClient, _parse_json_content
from app.config import Settings
from app.core.analysis_service import AnalysisService, ModelOutputError, _RawAssessment
from app.core.prompts import ANALYST_SYSTEM_PROMPT
from app.core.models import AccountAnalysisRequest, MonthlySummaryRow
from app.core.summary_loader import SummaryCsvError, parse_monthly_summary_csv


class ImmediateQueue:
    async def submit(self, name, operation):
        return await operation()


class FakeLlm:
    def __init__(self, response):
        self.response = response

    async def complete_json(self, **_kwargs):
        return self.response


def row(month: str, count: int) -> MonthlySummaryRow:
    return MonthlySummaryRow(
        acct_num="4110249230", year_month=month, txn_count_monthly=count,
        pct_burst=0, total_amount=100, avg_amount=100, std_amount=0,
        max_amount=100, pct_trx_gap=1, monthly_debit=50, monthly_credit=50,
        debit_count_monthly=1, credit_count_monthly=1,
        monthly_avg_debit=50, monthly_avg_credit=50,
    )


class StrictAssessmentTests(unittest.TestCase):
    def test_uppercase_summary_headers_are_parsed(self):
        header = (
            "YEAR_MONTH,ACCT_NUM,txn_count_monthly,debit_count_monthly,credit_count_monthly,"
            "total_amount,avg_amount,std_amount,max_amount,pct_burst,pct_trx_gap,monthly_debit,"
            "monthly_credit,monthly_avg_debit,monthly_avg_credit"
        )
        rows = [
            f"20260{month},4110249230,1,1,0,100,100,0,100,0,1,100,0,100,0"
            for month in range(1, 7)
        ]
        parsed = parse_monthly_summary_csv("\n".join([header, *rows]))
        self.assertEqual([item.year_month for item in parsed], [f"20260{m}" for m in range(1, 7)])

    def test_nonconsecutive_months_are_rejected(self):
        header = (
            "year_month,acct_num,txn_count_monthly,debit_count_monthly,credit_count_monthly,"
            "total_amount,avg_amount,std_amount,max_amount,pct_burst,pct_trx_gap,monthly_debit,"
            "monthly_credit,monthly_avg_debit,monthly_avg_credit"
        )
        months = ["202601", "202602", "202603", "202604", "202605", "202607"]
        content = "\n".join([header, *[
            f"{month},4110249230,1,1,0,100,100,0,100,0,1,100,0,100,0" for month in months
        ]])
        with self.assertRaises(SummaryCsvError):
            parse_monthly_summary_csv(content)

    def test_exact_evidence_is_hydrated_and_sets_risk(self):
        response = {
            "monthly_comparison": [{
                "title": "Activity resumed", "pattern_summary": "Activity followed four inactive months.",
                "evidence_ids": ["M202605.txn_count_monthly", "M202606.txn_count_monthly"],
            }],
            "profile_timeline_notes": [],
            "findings": [{
                "category": "dormancy_reactivation", "severity": "medium",
                "rationale": "The account resumed activity after consecutive inactive months.",
                "evidence_ids": ["M202604.txn_count_monthly", "M202605.txn_count_monthly"],
            }],
            "executive_summary": "A material change in activity requires review.",
            "limitations": [],
        }
        request = AccountAnalysisRequest(
            case_id="case-1", source_filename="summary.csv", acct_num="4110249230",
            monthly_summary=[row(f"20260{month}", 0 if month < 5 else 1) for month in range(1, 7)],
        )
        service = AnalysisService(FakeLlm(response), Settings(), ImmediateQueue())
        assessment = asyncio.run(service.analyze_account(request))
        self.assertEqual(assessment.risk_level, "medium")
        self.assertEqual(assessment.findings[0].evidence[0].value, "0")
        self.assertEqual(assessment.findings[0].evidence[0].feature, "txn_count_monthly")

    def test_unknown_evidence_fails_closed(self):
        response = {
            "monthly_comparison": [], "profile_timeline_notes": [],
            "findings": [{
                "category": "activity_spike", "severity": "high", "rationale": "Unsupported source reference.",
                "evidence_ids": ["M202699.monthly_credit"],
            }],
            "executive_summary": "Unsupported.", "limitations": [],
        }
        request = AccountAnalysisRequest(
            case_id="case-2", source_filename="summary.csv", acct_num="4110249230",
            monthly_summary=[row(f"20260{month}", 1) for month in range(1, 7)],
        )
        service = AnalysisService(FakeLlm(response), Settings(), ImmediateQueue())
        with self.assertRaises(ModelOutputError):
            asyncio.run(service.analyze_account(request))

    def test_invalid_or_extra_json_is_not_repaired(self):
        with self.assertRaises(LlmServiceError):
            _parse_json_content('{"findings": []')
        with self.assertRaises(LlmServiceError):
            _parse_json_content('{"findings": []} explanation')
        self.assertEqual(_parse_json_content('{"findings": []}'), {"findings": []})

    def test_request_has_real_system_role_and_schema_mode(self):
        client = OpenAICompatibleClient(Settings())
        body = client._build_body("SYSTEM", {"case": "x"}, use_response_format=True, response_schema={"type": "object"})
        self.addCleanup(lambda: asyncio.run(client.close()))
        self.assertEqual(body["messages"][0], {"role": "system", "content": "SYSTEM"})
        self.assertEqual(body["messages"][1]["role"], "user")
        self.assertNotIn("stop", body)
        self.assertNotIn("strict", body["response_format"]["json_schema"])

    def test_output_schema_avoids_date_time_format_backend_incompatibility(self):
        schema = _RawAssessment.model_json_schema()
        self.assertNotIn('"format": "date-time"', json.dumps(schema))

    def test_prompt_teaches_semantic_rubric_without_copyable_json_examples(self):
        self.assertIn("dormancy_reactivation", ANALYST_SYSTEM_PROMPT)
        self.assertIn("Read all six txn_count_monthly values", ANALYST_SYSTEM_PROMPT)
        self.assertNotIn("Example 1", ANALYST_SYSTEM_PROMPT)


if __name__ == "__main__":
    unittest.main()
