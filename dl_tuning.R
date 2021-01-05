getwd()
setwd("Z:/X-Drive/Common/MHasan/ML")

ibrary(h2o)
library('stringr')


h2oServer <- h2o.init(ip="localhost", port=54321, max_mem_size="4g", nthreads=-1)
#h2oServer <- h2o.init(ip="h2o-cluster", port=54321) # optional: connect to running H2O cluster


train_hex <- h2o.importFile(h2oServer, path = "https://s3.amazonaws.com/h2o-training/mnist/train.csv.gz",
                            header = F, sep = ',', dest = 'train.hex')
test_hex <- h2o.importFile(h2oServer, path = "https://s3.amazonaws.com/h2o-training/mnist/test.csv.gz",
                           header = F, sep = ',', dest = 'test.hex')


train_hex[,785] <- as.factor(train_hex[,785])
test_hex[,785] <- as.factor(test_hex[,785])
