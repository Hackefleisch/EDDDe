# WelQrate

### How to read the activity_value column?

(Update on June 26, 2024)
 
Our benchmark currently has 3 regression datasets: AID435008, AID2689, and AID488997.

1. For AID435008: 
The regression data was extracted from AIDs: [504699, 504701].
The IC50 values were averaged from those assays (given their identical experimental condition). 
The final IC50 data exported for 247 unique compounds, parsable from column "activity_value" in the .csv files with unit uM.
The range of IC50 values reported were from 0.24403999999999998 to 42.0585.
For final inactives, the IC50 value was set to 1000 uM if dose-response info was not available. This value could be modified by users.
The cut-off for activity was 10 uM. 
Assay information was retrieved from the PubChem database from the following URL(s):
https://pubchem.ncbi.nlm.nih.gov/bioassay/504699
https://pubchem.ncbi.nlm.nih.gov/bioassay/504701

2. For AID2689: 
The regression data was extracted from AIDs: [2821].
The final EC50_uM data exported for 224 unique compounds, parsable from column "activity_value" in the .csv files with unit uM.
The range of EC50_uM values reported were from 0.1 to 120.0.
For final inactives, the EC50_uM value was set to 1000 uM if dose-response info was not available. This value could be modified by users.
The cut-off for activity was not reported ("IC50 > 1 log over the highest tested concentration being active").
Assay information was retrieved from the PubChem database from the following URL(s):
https://pubchem.ncbi.nlm.nih.gov/bioassay/2821

3. For AID488997:
The regression data was extracted from AIDs: [588401, 504840].
The IC50 values were averaged from those assays (given their identical experimental condition). 
The final IC50 data exported for 896 unique compounds, parsable from column "activity_value" in the .csv files with unit uM.
The range of IC50 values reported were from 4.17e-05 to 30.1995.
For final inactives, the IC50 value was set to 1000 uM if dose-response info was not available. This value could be modified by users.
The cut-off for activity was not reported.
Assay information was retrieved from the PubChem database from the following URL(s):
https://pubchem.ncbi.nlm.nih.gov/bioassay/588401
https://pubchem.ncbi.nlm.nih.gov/bioassay/504840



## Standard Data Format (CSV)

### Naming convention

All CSV files are named in the format {PubChem Assay Indentifier}_{Activity Outcome}.
For example, AID488997_actives contains all final active molecules in the dataset AID488997. On the other hand, AID488997_inactives contains all final inactive molecules. 

### How to read our CSV data files?

Each CSV file contains a datatable with 8 columns: 

0. CID: Compound ID, or accession CID for compounds. For information on Compound and Compound ID, refer to https://pubchem.ncbi.nlm.nih.gov/docs/compounds. 

1. SMILES: Isomeric SMILES (Simplified molecular-input line-entry system) representations for compounds. 

2. InChI: Standard InChI (International Chemical Identifier) representations for compounds.

3. activity_outcome: assay result for compounds (binary: Active/Inactive). 

4. activity_value: the dose-response assay activity values for compounds (uM). Should be empty for classification-only datasets. For more details on how to understand these values, refer to the section below. 

5. mol_removed_from_mixture: A certain number of mixtures were processed in our curation pipeline based on the rules defined in our main paper. In particular, a major component of the mixture might be retained, and the other minor component(s) might be removed. If such situation happens, the SMILES string of the removed component(s) will be recorded in this column for user reference, while the retained component will replace the orignal mixture in column #1 and #2. Since there are multiple ways of handling mixtures, user could decide if they would like to proceed with the paticular processing of such compounds or prefer not to use the processed compounds in their benchmark. 

6. small_inorganic_mol_from_mixture: If the removed component(s) (in column #5) is an inorganic compound(s), its SMILES will also appear in this column. 

7. small_organic_mol_from_mixture: If the removed component(s) (in column #5) is an organic compound(s), its SMILES will also appear in this column. 




