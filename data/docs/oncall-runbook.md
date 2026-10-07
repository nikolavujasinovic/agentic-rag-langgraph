# On-Call and Incident Response Runbook

## Severity levels
SEV1: full outage or data loss affecting many customers. SEV2: major feature degraded for many
customers. SEV3: minor issue or single-customer impact with a workaround.

## When a SEV1 is declared
1. The on-call engineer acknowledges the page within 5 minutes.
2. Open a dedicated incident channel named #inc-YYYYMMDD-short-name.
3. Assign an Incident Commander (IC); the on-call engineer is IC until relieved.
4. Post a status page update within 15 minutes and every 30 minutes afterwards.
5. Notify the VP of Engineering by phone.

## After the incident
A blameless postmortem is required for every SEV1 and SEV2 within 5 business days. Action items
are tracked in the INCIDENT Jira project with an owner and due date.

## On-call rotation
Rotations are one week long, handed over on Mondays at 10:00 CET. Engineers receive an on-call
stipend and a compensatory day off after any night page.
