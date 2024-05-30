import json

from pyspark.sql import SparkSession


class SparkJsonReader():
   def __init__(self) -> None:
      self.spark = self.get_spark_session()

   def get_spark_session(self):
      spark = SparkSession.builder \
            .master("local[*]") \
            .config("spark.io.compression.codec", "snappy") \
            .config("spark.ui.enabled", "false") \
            .config("spark.executor.memory", "4g") \
            .config("spark.driver.memory", "4g") \
            .config("spark.memory.offHeap.enabled", True) \
            .config("spark.memory.offHeap.size", "4g") \
            .appName("sampleCodeForReference") \
            .getOrCreate()
      return spark
    
   def read_json(self,path):
      with open(path, "r") as file:
        data = json.load(file)
      return data
   
   def flatten_json(self, data):
      keys = list(data.keys())

      flattened_data = []
      for idx in range(len(data[keys[0]].keys())):
        document = {}
        for key in keys:
            document[key] = data[key][str(idx)]
        flattened_data.append(document) 

      return flattened_data
   
   def create_spark_df(self,path):
      data = self.read_json(path)
      flattened_json = self.flatten_json(data)
      json_df = self.spark.createDataFrame(flattened_json)
      return json_df
      
 
      
            
        


if __name__ == '__main__':
   spark_read_json = SparkJsonReader()
   json_df = spark_read_json.create_spark_df('/home/kedhar/Downloads/imdb_processed.json')
   print(json_df.show())





