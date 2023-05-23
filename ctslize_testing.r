#load some required libraries
library(RCurl)
library(glmnet)
library(xgboost)
library(h2o)
library(caret)
#library(HDeconometrics)
#library(varbvs)

custfit.lasso = function(vect)
{
  lasso=ic.glmnet(x = vect$x,y=vect$y,family = "gaussian",alpha = 1)
  
  return(list(lasso = lasso,vars =  as.integer(lasso$glmnet$beta[,lasso$glmnet$dim[2]]!=0)))
}
#define a custom function 1 to predict
custpredict.lasso = function(infer, x.new)
{
  return(predict(infer$lasso$glmnet,newx = x.new,type = "response")[,which(infer$lasso$glmnet$lambda == infer$lasso$lambda)])#
}
#define a custom function 2 to fit
custfit.ridge = function(vect)
{
  ridge = ic.glmnet(x = vect$x,y=vect$y,family = "gaussian",alpha = 0)
  return(list(ridge = ridge, vars = as.integer(ridge$glmnet$beta[,ridge$glmnet$dim[2]]!=0)))
}
#define a custom function 2 to predict
custpredict.ridge = function(infer, x.new)
{
  return(predict(infer$ridge$glmnet,newx = x.new,type = "response")[,which(infer$ridge$glmnet$lambda == infer$ridge$lambda)])
}


#vect = NULL
#vect$x = as.matrix(data.example[,-1])
#vect$y = data.example$M

library(forecast)
library(glmnet)
#library(HDeconometrics)
#library(BAS)
library(xgboost)
#library(varbvs)


custfit.lasso = function(vect)
{
  lasso=ic.glmnet(x = vect$x,y=vect$y,family = "gaussian",alpha = 1)
  
  return(list(lasso = lasso,vars =  as.integer(lasso$glmnet$beta[,lasso$glmnet$dim[2]]!=0)))
}
#define a custom function 1 to predict
custpredict.lasso = function(infer, x.new)
{
  return(predict(infer$lasso$glmnet,newx = x.new,type = "response")[,which(infer$lasso$glmnet$lambda == infer$lasso$lambda)])#
}
#define a custom function 2 to fit
custfit.ridge = function(vect)
{
  ridge = ic.glmnet(x = vect$x,y=vect$y,family = "gaussian",alpha = 0)
  return(list(ridge = ridge, vars = as.integer(ridge$glmnet$beta[,ridge$glmnet$dim[2]]!=0)))
}
#define a custom function 2 to predict
custpredict.ridge = function(infer, x.new)
{
  return(predict(infer$ridge$glmnet,newx = x.new,type = "response")[,which(infer$ridge$glmnet$lambda == infer$ridge$lambda)])
}


custfit.vb = function(vect)
{
  vb = varbvs(X = vect$x,y=vect$y,Z=vect$x,family = "gaussian",verbose = FALSE, maxiter = 1000)
  vars = as.integer(vb$pip>=0.5)
  return(list(vb = vb, vars = vars))
}
#define a custom function 3 to predict
custpredict.vb = function(infer, x.new)
{
  return(predict(infer$vb,X = x.new,Z=x.new))
} 

#define your working directory, where the data files are stored
workdir=""


data.example = read.csv("slice_localization.csv",sep=",", header = T)


#data.example$MS=as.integer(data.example$V1=="M")
#data.example$FS=as.integer(data.example$V1=="F")
#data.example$V1=data.example$V9
#data.example$V9 = NULL

#names(data.example) = c("reference","Length", "Diameter","Height","WholeWeight","ShuckedWeight","VisceraWeight","ShellWeight","Male","Femele")

set.seed(040590)
teid =  read.csv("teids_ct.csv", header = F)[,1]


test = data.example[teid,]
data.example = data.example[-teid,]
train = data.example


vect = NULL
vect$x = as.matrix(data.example[,-ncol(data.example)])
vect$y = data.example$reference

sum(test$reference)


gc()

#prepare the data structures for the final results
results=array(0,dim = c(11,100,5))

# h2o initialize
h2o.init(nthreads=-1, max_mem_size = "6G")
h2o.removeAll()
# h2o.random forest
df = as.h2o(data.example)

train1 = h2o.assign(df , "train1.hex")
valid1 = h2o.assign(df , "valid1.hex")
test1 = h2o.assign(as.h2o(test[,-ncol(data.example)]), "test1.hex")
features = names(train1)[-ncol(data.example)]
for(ii in 1:10)
{
  print(paste("iteration ",ii))
  #here we are no longer running BGNLM, since BGNLM algorithms are run via other scripts
  #for computational efficiency and speed
  # capture.output({withRestarts(tryCatch(capture.output({
    #run xGboost logloss gblinear
    t=system.time({
      param = list(objective = "reg:squarederror",
                   eval_metric = "rmse",
                   booster = "gblinear",
                   eta = 0.05,
                   subsample = 0.86,
                   colsample_bytree = 0.92,
                   colsample_bylevel = 0.9,
                   min_child_weight = 0,
                   gamma = 0.005,
                   max_depth = 15)
      
      
      dval=xgb.DMatrix(data = data.matrix(train[,-ncol(data.example)]), label = data.matrix(train[,ncol(data.example)]),missing=NA)
      watchlist=list(dval=dval)
      
      
      m2 = xgb.train(data = xgb.DMatrix(data = data.matrix(train[,-ncol(data.example)]), label = data.matrix(train[,ncol(data.example)]),missing=NA),
                     param, nrounds = 5000,
                     watchlist = watchlist,
                     print_every_n = 10)
      
    })
    # Predict
    results[3,ii,4]=t[3]
    t=system.time({
      dtest  = xgb.DMatrix(data.matrix(test[,-ncol(data.example)]),missing=NA)
    })
    
    
    t=system.time({
      out = predict(m2, dtest)
    })
    results[3,ii,5]=t[3]
    
  
    
    #compute and store the performance metrics
    results[3,ii,1]= sqrt(mean((out - test$reference)^2))
    results[3,ii,2]=mean(abs(out - test$reference))
    results[3,ii,3] = cor(out,test$reference)
    
    # xgboost logLik gbtree
    t=system.time({
      param = list(objective = "reg:squarederror",
                   eval_metric = "rmse",
                   booster = "gbtree",
                   eta = 0.05,
                   subsample = 0.86,
                   colsample_bytree = 0.92,
                   colsample_bylevel = 0.9,
                   min_child_weight = 0,
                   gamma = 0.005,
                   max_depth = 15)
      
      dval=xgb.DMatrix(data = data.matrix(train[,-ncol(data.example)]), label = data.matrix(train[,ncol(data.example)]),missing=NA)
      watchlist=list(dval=dval)
      
      
      m2 = xgb.train(data = xgb.DMatrix(data = data.matrix(train[,-ncol(data.example)]), label = data.matrix(train[,ncol(data.example)]),missing=NA),
                    param, nrounds = 300,
                    watchlist = watchlist,
                    print_every_n = 10)
      
    })
    
    results[4,ii,4]=t[3]
    # Predict
    system.time({
      dtest  = xgb.DMatrix(data.matrix(test[,-ncol(data.example)]),missing=NA)
    })
    
    t=system.time({
      out = predict(m2, dtest)
    })
    
    #compute and store the performance metrics
    #compute and store the performance metrics
    results[4,ii,1]= sqrt(mean((out - test$reference)^2))
    results[4,ii,2]=mean(abs(out - test$reference))
    results[4,ii,3] = cor(out,test$reference)
    
    
    
    #GLMNET (elastic networks) # lasso a=1
    t=system.time({
      infer.lasso = custfit.lasso(vect)
    })
    results[5,ii,4]=t[3]
    
    #predict
    t=system.time({
      out = custpredict.lasso(infer.lasso,as.matrix(test[,-ncol(data.example)]))
    })
    results[5,ii,5]=t[3]
    
    #compute and store the performance metrics
    results[5,ii,1]= sqrt(mean((out - test$reference)^2))
    results[5,ii,2]=mean(abs(out - test$reference))
    results[5,ii,3] = cor(out,test$reference)
    
    #ridge a=0
    t=system.time({
      infer.ridge = custfit.ridge(vect)
    })
    results[6,ii,4]=t[3]
    
    #predict
    t=system.time({
      out = custpredict.ridge(infer.ridge,as.matrix(test[,-ncol(data.example)]))
    })
    
    #compute and store the performance metrics
    results[6,ii,1]= sqrt(mean((out - test$reference)^2))
    results[6,ii,2]=mean(abs(out - test$reference))
    results[6,ii,3] = cor(out,test$reference)
    
    gc()
    
    
    
    #h2o naive bayes
    t=system.time({
      infer.vb = custfit.vb(vect)
    })
    #predict
    results[10,ii,4]=t[3]
    t=system.time({
      out=custpredict.vb(infer.vb,as.matrix(test[,-ncol(data.example)]))
    })
    results[10,ii,5]=t[3]

    
    #compute and store the performance metrics
    results[10,ii,1]= sqrt(mean((out - test$reference)^2))
    results[10,ii,2]=mean(abs(out - test$reference))
    results[10,ii,3] = cor(out,test$reference)
    
    gc()
  
    

    
    
    t=system.time({
      rf1 = h2o.randomForest( stopping_metric = "RMSE",
                              training_frame = train1,
                              validation_frame = valid1,
                              x=features,
                              y="reference",
                              model_id = "rf1",
                              ntrees = 10000,
                              stopping_rounds = 3,
                              score_each_iteration = T,
                              ignore_const_cols = T,
                              seed = ii)
    })
    results[7,ii,4]=t[3]
    
    #predict
    t=system.time({
      out=h2o.predict(rf1,as.h2o(test1))[,ncol(data.example)]
    })
    
    out=as.data.frame(as.matrix(out))$predict
    #compute and store the performance metrics
    results[7,ii,1]= sqrt(mean((out - test$reference)^2))
    results[7,ii,2] = mean(abs(out - test$reference))
    results[7,ii,3] = cor(out,test$reference)
    
    

    #h2o deeplearning
    t=system.time({
      neo.dl = h2o.deeplearning(x = features, y = "reference",hidden=c(200,200,200,200,200,200),
                                distribution = "gaussian",
                                training_frame = train1,
                                validation_frame = valid1,
                                seed = ii)
    })
    #predict
    t=system.time({
      out=h2o.predict(neo.dl,as.h2o(test1))[,ncol(data.example)]
    })
    results[8,ii,5]=t[3]
    out=as.data.frame(as.matrix(out))$predict
    results[8,ii,1]= sqrt(mean((out - test$reference)^2))
    results[8,ii,2] = mean(abs(out - test$reference))
    results[8,ii,3] = cor(out,test$reference)
    
    
    #h2o glm
    t=system.time({
      neo.glm = h2o.glm(x = features, y = "reference",
                        family = "gaussian",
                        training_frame = train1,
                        validation_frame = valid1,
                        #lambda = 0,
                        #alpha = 0,
                        lambda_search = F,
                        seed = ii)
    })
    #predict
    results[9,ii,4]=t[3]
    
    t=system.time({
      out=h2o.predict(neo.glm,as.h2o(test1))[,ncol(data.example)]
    })
    results[9,ii,5]=t[3]
    out=as.data.frame(as.matrix(out))$predict
    results[9,ii,1]= sqrt(mean((out - test$reference)^2))
    results[9,ii,2] = mean(abs(out - test$reference))
    results[9,ii,3] = cor(out,test$reference)
    
    print( results[,ii,1])
    
    gc()
  #})), abort = function(){onerr=TRUE;out=NULL})})
}

ids=1:100

ress=results[,ids,]

#make the joint summary of the runs, including min, max and medians of the performance metrics
summary.results=array(data = NA,dim = c(15,15))
for(i in 3:10)
{
  for(j in 1:5)
  {
    summary.results[i,(j-1)*3+1]=min(ress[i,,j])
    summary.results[i,(j-1)*3+2]=median(ress[i,,j])
    summary.results[i,(j-1)*3+3]=max(ress[i,,j])
  }
}
summary.results=as.data.frame(summary.results)

summary.results = summary.results[3:10,]

names(summary.results)=c("min(rmse)","median(rmse)","max(rmse)","min(mae)","median(mae)","max(mae)","min(corr)","median(corr)","max(cor)","min(ltime)","median(ltime)","max(ltime)","min(ptime)","median(ptime)","max(ptime)")
rownames(summary.results)=c("lXGBOOST(logLik)","tXGBOOST(logLik)","LASSO","RIDGE","RFOREST","DEEPNETS","GR","VARBAYESS")

write.csv(x = summary.results,file = "summarycompete.csv")

#write the final reults into the files
#write.csv(x = train,file = "/mn/sarpanitu/ansatte-u2/aliaksah/abeldata/breast cancer/train.csv")
#write.csv(x = test,file = "/mn/sarpanitu/ansatte-u2/aliaksah/abeldata/breast cancer/test.csv")