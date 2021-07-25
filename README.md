# Compressive Autoencoder for Image

Autoencoder usage for image compression, dimension reduction of datasets. Tensorflow api implementation for compression.

[Complete report](https://github.com/hsynercn/Compressive-Autorencoder-for-Image/blob/master/Report/report.pdf)

Autoencoders are primitive network models which generates the input in output layer, simply repeats the input. This function can decrease the noise in data or complete a missing part. Autoencoders extracts the attributes of data and uses same attributes for reconstruction phase. Extracted attributes can be used for reconstruction on an identical network in another location. We can use extracted attributes to compress data.

For compression model we need the middle layer outputs. These values forms the compressed data. At this point quantization problem emerges. Middle layer output of the network is very valuable and precise for reconstruction performance. Small changes can lead influential output errors. To minimize this problem precision guarded with 19 digit and exponentiation number. Network sigmoid transfer function generates outputs between 0.0 and 1.0. Python numpy.ndarray (n dimensional array) provides required function to transfer data to text files. Binary file recovery can create problem between big endian and little endian machines.

![image](https://user-images.githubusercontent.com/28985966/126912206-d98ceb78-7ff8-4d07-886d-017084945bed.png)
