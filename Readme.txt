Il file da eseguire per segmentare la immagini è Submission.ipynb, ovvero uno script Google Colab.




Per complettezza si sono poi riportati gli altri script usati nella fase di creazione e test del sistema. In particolare:

- allenamento_rete_5.py: codice che gestisce l'allenamento della rete
- analisi_risultati_rev5.py: script che prende il file contenente le metriche (creato da valutazione_rev5.py), creando un file che ordine automaticamente le 
			     reti dalla migliore alla peggiore e dei grafici relativi agli n tentativi migliori
- creazione_dataset.py: script che si occupa della divisione in training, validation e test set
- parametri_postprocess_rev5.py: script che si occupa di testare le varie combinazioni dei parametri del 
				  postprocess per individuare le migliori
- valutazione_rev5.py: script che crea un file contenente il valore di dice, errori e relative 
		       deviazioni standard

Sono poi presenti due file necessari al funzionamento di NNI in fase di training:

- config.yml: file contenente varie impostazioni relative a NNI, dove per esempio si definisce 
  `	      l'assessor per la strategia di early stopping
- search_space.json: file contenente le variabili da variare nel corso di un esperimento
