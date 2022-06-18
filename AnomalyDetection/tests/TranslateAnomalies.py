from reader.ODINTSTimeSeriesReader import ODINTSTimeSeriesReader

odin_reader = ODINTSTimeSeriesReader("data/dataset/anomalies_House1.csv",
									 "Time",
									 "Appliance1",
									 "start_date",
									 "end_date")
odin_reader.read("data/dataset/House1.csv")
df = odin_reader.get_dataframe()

print(df.iloc[82340:82360])
