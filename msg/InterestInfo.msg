Header header 

# The interestingness level, probabilities are in the range [0, 1].
float32 level

# The location of interestingness
geometry_msgs/Pose pose

# The image processed
float32[] image
int32[] image_shape

# The information of feature map
float32[] feature
int32[] feature_shape

# The reading weights from visual memory
float32[] reading_weights