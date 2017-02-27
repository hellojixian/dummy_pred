import DataLoader.MinutesDataLoader as DataLoader
import Common.config as config

if __name__ == "__main__":
    print("\n\nImporting data from path: {0} \n\n".format(config.MINUTE_DATA_PATH))
    DataLoader.load_data_to_db()