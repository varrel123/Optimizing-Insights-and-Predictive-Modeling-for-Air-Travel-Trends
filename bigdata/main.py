from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType

# Inisialisasi sesi Spark
spark = SparkSession.builder.appName("BigDataAnalysis").config("spark.sql.debug.maxToStringFields", 1000).getOrCreate()

# Skema dari Clean_Dataset
schema = StructType([
  StructField("_c0", IntegerType()),
  StructField("airline", StringType()),
  StructField("flight", StringType()),
  StructField("source_city", StringType()),
  StructField("departure_time", StringType()),
  StructField("stops", StringType()),
  StructField("arrival_time", StringType()),
  StructField("destination_city", StringType()),
  StructField("class", StringType()),
  StructField("duration", DoubleType()),
  StructField("days_left", IntegerType()),
  StructField("price", IntegerType())
])

# Membaca file CSV ke dalam DataFrame
df = spark.read.format("csv") \
  .option("header", True) \
  .option("delimiter", ",") \
  .option("quote", '"') \
  .schema(schema) \
  .load("Clean_Dataset.csv")

# Path untuk menyimpan output
output_file_path = "C:/Users/moham.BRAMASTA.001/PycharmProjects/bigdata/output_Clean_Dataset.txt"

# Mengarahkan output ke file output
import sys
original_stdout = sys.stdout
with open(output_file_path, 'w') as f:
    sys.stdout = f

# Analisis dan mengelola file pada Clean_Dataset.csv

    # 1. Menampilkan 10 baris pertama dari DataFrame
    print("\n5 baris pertama dari Clean_Dataset.csv:")
    df.show(5)

    # 2. Menampilkan skema DataFrame
    print("Skema DataFrame:")
    df.printSchema()

    # 3. Menghitung jumlah baris dalam DataFrame
    row_count = df.count()
    print(f"Jumlah Baris:\n{row_count}")
    print("\n")

    # 4. Menampilkan statistik deskriptif untuk kolom numerik
    numeric_summary = df.describe().toPandas()
    print("Statistik Deskriptif untuk Kolom Numerik:")
    print(numeric_summary)

    # 5. menghitung berapa banyak class economy dan business
    print("Table untuk menghitung berapa banyak kelas ekonomi dan bisnis")
    category_counts = df.groupBy("class").count().show()

    # 6. Melakukan filtering data duration lebih dari 3
    print("Table yang menampilkan Duration yang lebih besar dari 3")
    filtered_data = df.filter(col("duration") > 3).show()

    # 7. Pemodelan Mesin (Logistic Regression)
    print("Table yang menampilkan 10 baris pertama dari DataFrame yang sudah di-assembly")
    feature_columns = ["duration", "days_left", "price"]
    assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
    df_assembled = assembler.transform(df)
    df_assembled.show(10, False)

    # 8. Membuat String Indexer
    print("Table yang menampilkan 10 baris pertama dari DataFrame yang sudah di-index")
    indexer = StringIndexer(inputCol="class", outputCol="label")
    indexed = indexer.fit(df_assembled).transform(df_assembled)
    indexed.show(10, False)

# Mengembalikan output standar ke keadaan semula
sys.stdout = original_stdout
print(f"Output telah disimpan di: {output_file_path}")

# Menyimpan hasil analisis ke dalam file output
output_file_path = "path/to/output_Clean_Dataset.txt"


