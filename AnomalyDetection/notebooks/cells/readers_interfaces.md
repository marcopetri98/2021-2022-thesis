# Interfaces for data readers

Since data must be read from file, all types of data readers share some properties: they read from file, they return the dataset as a dataframe and they perform splitting of the dataset. Therefore, here it comes the need of abstracting interfaces for such classes. The interfaces define methods which will be public or at most protected.