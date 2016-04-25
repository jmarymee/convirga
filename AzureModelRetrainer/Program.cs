using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;

namespace AzureModelRetrainer
{
    class Program
    {
        static void Main(string[] args)
        {
            //These are all the properties necessary to perform model retraining. 
            //See the github repo for graphics and details 
            //See https://github.com/jmarymee/convirga
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

            //You can change the date run of the SQL query in order to get a daily trained model
            //retrainer.UpdateSQLQueryForNewDate("2015-10-01");

            //Upload new training set
            retrainer.UploadNewTrainingFileToStorage(@"C:\Users\jd\Downloads\AIC-Dataset.csv");

            //Used to get a blob handle to the retraining blob. Not necessary IF you uploaded a fresh set. If you did that then behind the scenes the lib grabbed a reference to the blob
            //But if you ALREADY uploaded a new set to your configured blob storage (using another tool) then this can be used to grab the Azure blob handle the retrainer needs. 
            retrainer.GetRetrainingBlob("AIC-Dataset.csv");

            //This is for display. it allows a person to view the results of the last model training.
            //This List is stored initially in the Library upon object instatiation. If there is no previous retrainging then there will only be one entry in the Dictionary. 
            //This example is here only so that you can comoare the last result with the new result after retraining. 
            //Console.WriteLine("Results of last training: ");
            //foreach(var val in retrainer.lastScores)
            //{
            //    Console.WriteLine("Rating Name: {0} : Value: {1}", val.Key, val.Value.ToString());
            //}

            //Retraining a model takes two steps; queue up the job then start the job. One must save the jobID in order to start the job
            //STEP ONE: Assuming your config params are correct, you queue up a retraining job using this call. Be sure to save the jobID or else you won't be able to start
            //the job. If you don't start the job it will be automatically deleted after a few minutes (per the ML doc)
            //The Dictionary of parms setup below are used when you are 'steering' the retraining - such as algorithmn parms, SQL queryies etc. 
            Dictionary<string, string> gParms = new Dictionary<string, string>();
            gParms.Add("Fraction of rows in the first output dataset", "0.5");
            string jobID = retrainer.QueueRetrainingAsync(MLRetrainerLib.RetrainerLib.TRAINING_DATA_SOURCE.DATA_UPLOAD, gParms).Result;

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
            Dictionary<string, double> scores = retrainer.GetLatestRetrainedResults();
            foreach (var val in scores)
            {
                Console.WriteLine("Rating Name: {0} : Value: {1}", val.Key, val.Value.ToString());
            }
            string resultOfRetrain = retrainer.GetLatestRetrainedResults(true);
            Console.WriteLine(resultOfRetrain);

            Console.WriteLine("Now deploying the new model to the published endpoint...");

            //STEP FOUR: Check to see if the new model is more accurate. 
            //Here is where we compare the current result to the last result. In this scenario, we compare AUC and if we haven't at 
            //least seen a 20% improvement then we don't deploy the retrained model
            //There are seven values to review and the API currently only allows you to select one. In this case it's AUC. The second param indicates the
            //improvement amount that the retrained model should have in order to get TRUE back from the API call.

            bool isModelbetter = true; //Arbitrary for testing...
            //bool isModelbetter = retrainer.isUdpateModel("AUC", 0.02f);
            if (!isModelbetter)
            {
                Console.WriteLine("No need to update endpoint. Accuracy has not improved. Press a key to end");
                Console.ReadLine();
                return; //if the model isn't more accurate then terminate the app by returning
            }

            //STEP FIVE: Deploy the updated, retrained model if you like the scores
            //Here is where we deploy the model to the published endpoint IF the accuracy has met our hurdle
            //You use the same jobID that you used to start the job
            bool isUpdated = retrainer.UpdateModel(jobID).Result;
            if (isUpdated)
            {
                Console.WriteLine("Successful model retraining and endpoint deployment");
            }
            else
            {
                Console.WriteLine("Something went wrong updating the model");
            }

            Console.WriteLine("Process has completed. Press a key to end");
            Console.ReadLine();
        }
    }
}
