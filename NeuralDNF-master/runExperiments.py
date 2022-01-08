from GraphNeuralNet import GGNN, printOrLog, deltaToTimeString
import os, argparse, pickle, time, numpy as np
import tensorflow as tf
import DNFGen
import time
import scipy.stats as st
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def log_or_print(v):
    if v.lower() in ('l','log'):
        return True
    elif v.lower() in ('p','print'):
        return False
    else:
        raise argparse.ArgumentTypeError("Invalid Option Returned for log_or_print")
def runExperiments_SynthData():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    argParser = argparse.ArgumentParser(description='Test the Neural#DNF system')
    argParser.add_argument("dataDir",type=str,help="The directory from which labelled test data is loaded")
    argParser.add_argument("-communicationProtocol", type=int, default=2, metavar='',
                           help="Communication Protocol to use for testing (Default is 2). ")
    argParser.add_argument("-embeddingDim", type=int, default=128, metavar='',
                           help="The embedding size to use (Default 128)")
    argParser.add_argument('-numIter', type=int, default=8, metavar='', help="The number of message passing "
                                                                             "iterations to run (Default 8)")
    argParser.add_argument("-weightsDir",type=str,default=None,help="The directory from which to load "
                                                                    "network parameters", metavar='')
    argParser.add_argument("-measureRunTime",type=str2bool,default=True, metavar='',
                           help="Specify whether timing tests should be conducted (Default True)")
    argParser.add_argument("-widthBasedAnalysis", type=str2bool, default=False, metavar='',
                           help="Measure Precision With Respect to Width (Default False)")
    argParser.add_argument("-outputFileName", type=str, default="Results.p",metavar='',
                           help="Set a custom output file name (Default Results.p)")
    argParser.add_argument("-logOpt", type=log_or_print, default=True, metavar='',
                           help="Choose whether to log progress in text file (L, Default) or print on console (P)")
    # Parse the Arguments
    args = argParser.parse_args()
    nbIter = args.numIter
    if args.weightsDir is not None:
        paramLocation = args.weightsDir+"/values.ckpt"
    else:
        paramLocation = None
    dataDir = args.dataDir
    commProt = args.communicationProtocol
    embDim = args.embeddingDim
    measureRunTime = args.measureRunTime
    fileName = args.outputFileName
    widthMeas = args.widthBasedAnalysis
    log_opt = args.logOpt
    fileNameBackup = "Backup"+fileName
    GraphNet = GGNN(nbIterations=nbIter,communicationProtocol=commProt,embeddingDim=embDim,widthExtract=widthMeas) # Initialise the Graph Neural Net
    GraphNet.loadParamsSession(paramLocation=paramLocation)
    dataFiles = [dataDir + f for f in os.listdir(dataDir) if f.endswith(".p")]# Prepare data for loading
    thresholds = np.array([0.01,0.02,0.05,0.1,0.15,0.2])
    thresholdsComparable = np.expand_dims(np.array(thresholds).T,axis=-1)
    nbThresholds = thresholds.shape[0]
    counts = np.zeros(nbThresholds)
    # Measured Entities
    batchRunTimes = []
    batchSizes = []
    networkMus = []
    networkSigmas = []
    KLMMus = []
    AbsDiffs = []
    KLDivs = []
    nbFormulas = 0
    test_widths = [3, 5, 8, 13, 21, 34]
    if widthMeas:
        sess = tf.Session()
        observedWidths = []
        width_success_dict = {width: [0] * len(test_widths) for width in test_widths}
        width_total_trial_dict = {width: [0] * len(test_widths) for width in test_widths}
    else:
        observedWidths = None
        sess = None
        width_success_dict = None
        width_total_trial_dict = None
    #Start experiments
    print(" Thresholds: " + str(thresholds))
    try:
        logFile = open("test_progress_log.txt", "w")
        filePath = "test_progress_log.txt"
        logFile.write("Log File Started: \r\n")
        logFile.close()
        intermediate_results = open("intermediate_results_log.txt", "w")
        intermediate_resultsPath = "intermediate_results_log.txt"
        intermediate_results.write("Log File Started: \r\n")
        intermediate_results.close()
        start_time = time.time()
        for index, dataFile in enumerate(dataFiles):
            intermediate_time = time.time()
            printOrLog(deltaToTimeString(intermediate_time - start_time)+")" + str(dataFile)+"||" +
                       str(nbFormulas) + " formulas tested", log_opt, filePath)
            printOrLog(deltaToTimeString(intermediate_time - start_time) + ")" + str(dataFile), log_opt,
                       intermediate_resultsPath)
            printOrLog("-----------", log_opt, intermediate_resultsPath)
            if nbFormulas > 0:
                printOrLog(str(np.divide(counts, nbFormulas)), log_opt, intermediate_resultsPath)
                if widthMeas:
                    for width in test_widths:
                        printOrLog("Width " + str(width) + ":", log_opt, intermediate_resultsPath)
                        printOrLog(str(np.divide(width_success_dict[width], width_total_trial_dict[width])),
                                   log_opt, intermediate_resultsPath)
            with open(dataFile, "rb" ) as file:
                batchSet = pickle.load(file) # Load batch set
                # New: Only load session once
            for index, batch in enumerate(batchSet):
                nbC, posLit, disjConj, conjLit, approxKL= batch  # Load individual batch
                '''
                approxRA = []
                logError = np.log(1 + 0.1)
                Zscore = st.norm.ppf(1 - 0.05 / 2)
                sigma = logError/Zscore
                for index , j in enumerate(approxKL) :
                  approxKL[index][0] = np.log(approxKL[index][0])
                  approxKL[index][1] = sigma
                print("approxKL",approxKL)
                '''
                if widthMeas:
                    widths = GraphNet.extractWidthFromBatch(conjLit=conjLit, disjConj=disjConj)  # Additional Step
                else:
                    widths = None

                if measureRunTime: # Measure Time
                    tBefore = time.time()
                    logMeans, logVariances = GraphNet.forwardPass(nbConjunctions=nbC, posLitProbs=posLit,
                                                                  disjConj = disjConj, conjLit=conjLit,
                                                                createSession=False)
                    print("logMeans, logVariances",logMeans, logVariances)                                               
                    runTime = time.time() - tBefore
                else: # Don't measure time
                    runTime="Not Measured" # Won't be used, but to eliminate the pesky warning
                    logMeans, logVariances = GraphNet.forwardPass(nbConjunctions=nbC, posLitProbs=posLit,
                                                                  disjConj=disjConj, conjLit=conjLit,
                                                                  createSession=False)
                print("logMeans, logVariances",logMeans, logVariances) 
    except KeyboardInterrupt: # Make robust to interruption
        pass
        #print("Saving Results so far and quitting ... ")
        #book.save(fileName)
def computeKLDiv(networkLogMean, networkLogStDev, approxMean, approxStDev):
    return np.log(approxStDev / networkLogStDev) - 0.5 + np.divide(networkLogStDev**2 +
            (approxMean - networkLogMean)**2, 2*approxStDev**2)
def computeAbsDiff(networkMean, approxMean):
    return np.abs(approxMean - networkMean)
def saveExcelSheet(book,fileName):
    print("Saving Results so far and quitting... ")
    book.save(fileName)
if __name__ == "__main__":
    runExperiments_SynthData()
