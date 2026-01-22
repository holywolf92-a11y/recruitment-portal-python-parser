# 🔧 Passport Extraction Enhancement

## Update Summary

Enhanced the Python parser's AI prompt to extract **all passport information** from passport documents.

## What Was Added

### New Fields Extracted from Passports:

1. **`nationality`** - Nationality (e.g., "Pakistani", "Indian")
2. **`passport_expiry`** - Passport expiry date (format: DD-MM-YYYY or YYYY-MM-DD)
3. **`expiry_date`** - Alternative field for passport expiry
4. **`issue_date`** - Passport issue date
5. **`place_of_issue`** - Place where passport was issued (e.g., "Islamabad", "Karachi")
6. **`father_name`** - Father's name (if available on passport)
7. **`date_of_birth`** - Date of birth (improved format handling)

### Enhanced Prompt

The AI prompt now specifically asks for:
- Passport number (e.g., PA1234567, AB1234567)
- Nationality
- Passport expiry date
- Issue date
- Place of issue
- Date of birth (with multiple format support)

### Backward Compatibility

- Still extracts `dob` field for backward compatibility
- Maps `dob` to `date_of_birth` if `date_of_birth` is missing
- Maps `expiry_date` to `passport_expiry` if `passport_expiry` is missing

## Example Passport Data

**Input (from passport document):**
```
Full Name: Muhammad Farhan
Passport No: PA1234567
Nationality: Pakistani
Date of Birth: 15-08-1994
Issue Date: 10-06-2022
Expiry Date: 09-06-2032
Place of Issue: Islamabad
```

**Extracted Output:**
```json
{
  "category": "passport",
  "confidence": 0.98,
  "identity_fields": {
    "name": "Muhammad Farhan",
    "passport_no": "PA1234567",
    "nationality": "Pakistani",
    "date_of_birth": "15-08-1994",
    "passport_expiry": "09-06-2032",
    "expiry_date": "09-06-2032",
    "issue_date": "10-06-2022",
    "place_of_issue": "Islamabad"
  }
}
```

## Backend Integration

The backend worker (`documentVerificationWorker.ts`) now:
1. Maps Python parser response to `extracted_identity` format
2. Updates candidate records with extracted information:
   - `nationality` → `candidates.nationality`
   - `passport_no` → `candidates.passport_normalized` + `candidates.passport`
   - `passport_expiry` → `candidates.passport_expiry`
   - `date_of_birth` → `candidates.date_of_birth`

## Testing

1. Upload a passport document
2. Check Python parser logs for extracted fields
3. Check backend logs for candidate updates:
   ```
   [DocumentVerification] Updating nationality: Pakistani
   [DocumentVerification] Updating passport: PA1234567
   [DocumentVerification] Updating passport expiry: 2032-06-09
   [DocumentVerification] Updating date of birth: 1994-08-15
   ```
4. Verify Excel Browser shows updated information

## Deployment

After deploying this update:
1. Python parser will extract all passport fields
2. Backend will update candidate records automatically
3. Excel Browser will show complete information
