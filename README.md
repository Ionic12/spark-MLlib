# Description
<div align="justify">
Repository ini berisi implementasi tugas menggunakan Spark MLlib dengan melakukan percobaan yang serupa dengan yang dilakukan di CloudxLab. Repository ini dirancang untuk membantu pengguna mempelajari dan mempraktekkan Spark MLlib dengan contoh-contoh yang mirip dengan yang ada di CloudxLab.
</div>

## Folder & Data
<img src="folder.png" />
<img src="data.png" />
<img src="hasil 1.1.png" />

# Movie Lens Ratings
<div>
  <pre>
    <code>
import org.apache.spark.ml.recommendation.ALS
case class Rating(userId: Int, movieId: Int, rating: Float, timestamp: Long)
def parseRating(str: String): Rating = {
  val fields = str.split("::")
  assert(fields.size == 4)
  Rating(fields(0).toInt, fields(1).toInt, fields(2).toFloat, fields(3).toLong)
}
parseRating("1::1193::5::978300760")
var raw = sc.textFile("/data/ml-1m/ratings.dat")
raw.take(1)
val ratings = raw.map(parseRating).toDF()
ratings.show(5)
val Array(training, test) = ratings.randomSplit(Array(0.8, 0.2))
val als = new ALS().setMaxIter(5).setRegParam(0.01).setUserCol("userId").setItemCol("movieId").setRatingCol("rating")
val model = als.fit(training)
model.save("mymodel")
val predictions = model.transform(test)
predictions.map(r => r(2).asInstanceOf[Float] - r(4).asInstanceOf[Float])
.map(x => x*x)
.filter(!_.isNaN)
.reduce(_ + _)
predictions.take(10)
predictions.write.format("com.databricks.spark.csv").save("ml-predictions.csv")
    </code>
  </pre>
  <p align="justify">
  Apache Spark MLlib untuk membangun sistem rekomendasi dengan Collaborative Filtering menggunakan model ALS. Langkah-langkahnya meliputi mengurai data rating menjadi objek Rating, membaca data rating dari file, mengubahnya menjadi DataFrame, membagi data menjadi subset training dan test, melatih model ALS dengan data training, melakukan prediksi pada data test, menghitung error prediksi, dan menyimpan hasil prediksi ke dalam file CSV.
  </p>
</div>
<img src="step 1.1.png" />
<img src="step 1.2.png" />
<img src="step 1.3.png" />
