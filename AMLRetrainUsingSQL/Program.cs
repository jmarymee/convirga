using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AMLRetrainUsingSQL
{
    class Program
    {
        static void Main(string[] args)
        {
            //These are all the properties necessary to perform model retraining. 
            //See the github repo for graphics and details 
            //See https://github.com/jmarymee/azuremlmodelretrainer
            //
            //First we create an object to store all of the config parameters. 
            MLRetrainerLib.RetrainerLib.MLRetrainConfig configobj = new MLRetrainerLib.RetrainerLib.MLRetrainConfig();

            //We first add the parameters for the retrainer experiment. These are found in the Azure Studio. You must first publish your
            //training experiement as a retraining endpoint. Today is it called a 'predictive experiment' in Azure ML Studio
            //Once created it will have at least two output nodes; one for the newly updated/trained model (the iLearner) and a metrics/accuracy scores output
            //You can then publish as a web service endpoint and obtain the two following values:
            configobj.mlretrainerurl = Properties.Settings.Default.mlretrainermodelurl;
            configobj.mlretrainerkey = Properties.Settings.Default.mlretrainerkey;

            //In order to get these setup you MUST go to the Azure Portal (currently the legacy one), go to the ML workspace and go to Web Services. Go to the 
            //Predictive experiment endpoint (not the training one) and create another endpoint. The reason for this is that you cannot deploy a new model to the initial, default 
            //endpoint. Once you create the additional endpoint here, you can first locate the API key on the lower right side (that will be your endpoint key)
            //and then find and click on Update Resource. That will show you the endpoint URL. You can find the endpoint name in the sample C# code. BE SURE that you
            //copy it exactly as it is in the C# example. If it is mismatched, it will cause a failure when you attempt to programmatically deply the retrained model
            configobj.publishendpointurl = Properties.Settings.Default.enpointurl;
            configobj.publishendpointkey = Properties.Settings.Default.endpointkey;
            configobj.publishendpointname = Properties.Settings.Default.endpointname;


            //These are somewhat arbitrary. Essentially it's just a place to store the retrained models and CSV files with the scoring/accuracy daata. 
            //You should manually created this storage area in Azure Blobs and then use the url, key and container name in the params below.
            configobj.mlretrainerstoragename = Properties.Settings.Default.mlstoragename;
            configobj.mlretrainerstoragekey = Properties.Settings.Default.mlstoragekey;
            configobj.mlretrainercontainername = Properties.Settings.Default.mlstoragecontainer;


            //This only takes one object but it must be prefilled first based on the notes above. 
            //there are some internal activities that take place when you instantiate the object. One is that it will attempt to load
            //scoring data from the last time you ran this model retraining. If it doesn't exist, then it still creates an internal object but
            //it only has one entry in the Dictionary with a text string indicating no scoring data. 
            MLRetrainerLib.RetrainerLib retrainer = new MLRetrainerLib.RetrainerLib(configobj);

            //Retraining a model takes two steps; queue up the job then start the job. One must save the jobID in order to start the job
            //STEP ONE: Assuming your config params are correct, you queue up a retraining job using this call. Be sure to save the jobID or else you won't be able to start
            //the job. If you don't start the job it will be automatically deleted after a few minutes (per the ML doc)
            Dictionary<string, string> gParms = new Dictionary<string, string>();
            gParms.Add("Database query", "select * from TblVCCompanies");
            gParms.Add("Fraction of rows in the first output dataset", "0.7");
            gParms.Add("L1 regularization weight", "1");
            //string jobID = retrainer.QueueRetrainingAsync(MLRetrainerLib.RetrainerLib.TRAINING_DATA_SOURCE.CLOUD_HOSTED, "select top(2000) * from TblVCCompanies").Result;
            string jobID = retrainer.QueueRetrainingAsync(MLRetrainerLib.RetrainerLib.TRAINING_DATA_SOURCE.CLOUD_HOSTED, null).Result;

            //STEP TWO: This is how you start the retraining job
            retrainer.StartRetrainingJob(jobID).Wait();

            //We use this to watch the retraining so that we can decide if we want to deploy to the endpoint or not. 
            //We spin here on the token until we show complete
            //This var is declared here and checked in the Do/While below. Of course you could also use Delegates or Threads to wait on this if the main thread has other
            //work you want it do do. 
            MLRetrainerLib.BatchScoreStatusCode status;

            //STEP THREE: Wait for retraining completion
            //Here is when we spin lock until we show that the retraining job is Finished
            do
            {
                status = retrainer.CheckJobStatus(jobID).Result;
                Console.WriteLine(status.ToString());
            } while (!(status == MLRetrainerLib.BatchScoreStatusCode.Finished));

            //Now we look at the new (latest) results. 
            //These are pulled from CSV files in the configured Azure Blob Storaged
            Console.WriteLine("New Scores for retraining...");
            //Dictionary<string, double> scores = retrainer.GetLatestRetrainedResults();
            //foreach (var val in scores)
            //{
            //    Console.WriteLine("Rating Name: {0} : Value: {1}", val.Key, val.Value.ToString());
            //}

            //string resultOfRetrain = retrainer.GetLatestRetrainedResults(true);
            //Console.WriteLine(resultOfRetrain);
        }
    }
}
