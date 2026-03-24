Put datasets under data/raw and adjust loaders if needed.

This project expects to produce unified CSVs:
- data/processed/l1_anomaly.csv  columns: payload,label_anomaly
- data/processed/l2_threat.csv   columns: raw_text,label_type

If your raw datasets differ, edit parsing in:
- csic2010_loader.py
- httpparams_loader.py
- ecml_loader.py
- xss_loader.py
