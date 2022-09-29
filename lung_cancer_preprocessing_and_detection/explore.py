import time

import pylidc as pl

# Query for all CT scans with desired traits.
scans = pl.query(pl.Scan).first()
nods = scans.cluster_annotations()
scans.visualize(annotation_groups=nods)

print("Sleeping")
time.sleep(3)

anns = pl.query(pl.Annotation).first()
anns.visualize_in_scan()
