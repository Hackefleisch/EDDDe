The subdirectories contain random subsets of the described datasets. For the welQrate set all actives and random inactives were chosen.

SMRT has a continues variable, welQrate is a binary classification.

The csv file holds an ID, a SMILE molecular descriptor and a target value.

Not all entries in the csv file could be used to generate electron density coefficients, usually due to the presence of uncommon atom types.

All calculations were done using RDKit generated 3D conformers.

coefficients.pkl is a pickled file and holds a dictionary. Molecule IDs are the keys. The entries are a single matrix of dimension (number of atoms)*127

adjacency.pkl and distance.pkl follow the scheme, but their entries are matrices of dim (num atoms)*(num atoms). adjacency has 0 or 1 entries describing if two atoms are conntected via a bond. distance has the shortest path between two atoms in number of bonds.
