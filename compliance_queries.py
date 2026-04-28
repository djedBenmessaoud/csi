#!/usr/bin/env python3
"""
Compliance Query Axes - Predefined questions that drive the gap analysis.
"""

COMPLIANCE_AXES = [
    {
        "id": "deletion_rights",
        "label": "Right to deletion / erasure",
        "query": "Under what conditions can a person request deletion of their personal data, and what are the exceptions?"
    },
    {
        "id": "breach_notification",
        "label": "Data breach notification requirements",
        "query": "What are the mandatory timelines and procedures for notifying authorities and individuals after a data breach?"
    },
    {
        "id": "consent",
        "label": "Consent requirements for data processing",
        "query": "What are the requirements for obtaining valid consent before collecting or processing personal data?"
    },
    {
        "id": "data_portability",
        "label": "Right to data portability",
        "query": "What rights do individuals have to receive and transfer their personal data to another service?"
    },
    {
        "id": "access_rights",
        "label": "Right to access / know",
        "query": "What information must a company disclose when a person requests to know what personal data is collected about them?"
    },
    {
        "id": "opt_out_sale",
        "label": "Opt-out of data sale",
        "query": "What rights do individuals have to opt out of the sale or sharing of their personal data to third parties?"
    },
    {
        "id": "children_data",
        "label": "Children's data protections",
        "query": "What special protections and consent requirements apply to collecting data from minors or children?"
    },
    {
        "id": "penalties",
        "label": "Penalties and enforcement",
        "query": "What are the financial penalties and enforcement mechanisms for non-compliance?"
    }
]
