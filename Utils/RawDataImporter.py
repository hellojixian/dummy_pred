import DataLoader.MinutesDataLoader as DataLoader
import config

print("\n\nImporting data from path: {0} \n\n".format(config.MINUTE_DATA_PATH))
DataLoader.load_data_to_db()