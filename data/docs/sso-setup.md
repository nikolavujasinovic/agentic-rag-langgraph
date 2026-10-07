# SSO Setup Guide

## Supported identity providers
Acme Cloud supports SAML 2.0 single sign-on with Okta, Microsoft Entra ID and Google Workspace.
SAML SSO is available on the Enterprise tier; Pro includes Google Workspace SSO only.

## Configuration steps
1. In the Acme admin console open Settings > Security > SSO.
2. Copy the ACS URL and Entity ID into the identity provider's SAML application.
3. Upload the IdP metadata XML to Acme.
4. Test with a non-admin account before enforcing SSO for the whole organization.

## Our security defaults
Acme signs SAML requests with SHA-256 and requires signed assertions. IdP-initiated login is
enabled by default for convenience. Sessions expire after 12 hours of inactivity. User
deprovisioning is manual unless the customer enables SCIM (Enterprise only).

## Troubleshooting
Most failures are caused by clock skew greater than 3 minutes between the IdP and Acme, or an
Entity ID mismatch. Check the SSO audit log under Settings > Security > Logs.
