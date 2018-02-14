library("foreach")
library("doParallel")

registerDoParallel()
set.seed(100)

system.time(
            foreach (i = 1:4) %dopar% {
              a = replicate(10000, rnorm(10000))
              d = determinant(a)
            }
            )

doParallel::stopImplicitCluster()

