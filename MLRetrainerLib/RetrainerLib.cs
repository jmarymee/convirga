using Microsoft.WindowsAzure.Storage;
using Microsoft.WindowsAzure.Storage.Blob;
using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;

namespace MLRetrainerLib
{
    public class RetrainerLib
    {
        /// <summary>
        /// This property is the published ML retraing web service. You can find it on the Azure ML Studio page under published web services.
        /// Be sure to use the one from the training web service versus the scoring web service
        /// </summary>
        public string _mlretrainmodelurl { get; set; }
        public string _mlretrainerkey { get; set; }
        public string _endpointurl { get; set; }
        public string _endpointkey { get; set; }
        public string _mlstoragename { get; set; }
        public string _mlstoragekey { get; set; }
        public string _mlstoragecontainer { get; set; }
        public string _nameOfEndpoint { get; set; } //used to push an updated model to the web endpoint
        public string _endpoint2url { get; set; }
        public string _endpoint2key { get; set; }

        private string _storageConnectionString;

        public string sqlQueryForRetraining { get; set; }

        private string _currentModelTraining;
        public string CurrentModelTrainingName
        {
            get { return _currentModelTraining;  }
        }

        public enum TRAINING_DATA_SOURCE { DATA_UPLOAD, CLOUD_HOSTED };
        public enum RESULTS_TYPE { BINARYCLASS, MULTICLASS };

        private string retrainerPrefix = "retrainer-";

        private CloudBlockBlob _trainingBlob;

        /// <summary>
        /// These are located and stored in this var during object construction for comparison. 
        /// </summary>
        public Dictionary<string, double> lastScores { get; set; }

        /// <summary>
        /// This is filled in after we have retrained the model and then placed the values here. Note that after model retraining the old scores are overwritten
        /// </summary>
        public Dictionary<string, double> retrainedScores { get; set; }

        /// <summary>
        /// This is the constructor for the library. We lazily use a param list rather than specify parameters, but parameters might be a good revision.
        /// Note: This means you must have the correct number of parameters AND they mus be in the correct order.
        /// mlretrainerurl : mlrertainerkey : pubendpointurl : pubendpointkey : mlstoragename : mlstoragecontainer : pubendpointname
        /// </summary>
        /// <param name="configs"></param>
        public RetrainerLib(MLRetrainConfig configobj)
        {
            _mlretrainmodelurl = configobj.mlretrainerurl;
            _mlretrainerkey = configobj.mlretrainerkey;
            _endpointurl = configobj.publishendpointurl;
            _endpointkey = configobj.publishendpointkey;
            _mlstoragename = configobj.mlretrainerstoragename;
            _mlstoragekey = configobj.mlretrainerstoragekey;
            _mlstoragecontainer = configobj.mlretrainercontainername;
            _nameOfEndpoint = configobj.publishendpointname;
            _endpoint2url = configobj.publishendpoint2url;
            _endpoint2key = configobj.publishendpoint2key;

            //This is used as the general URL for accessing the Azure Storage blobs where the updated iLearner and result scores are stored
            _storageConnectionString = string.Format("DefaultEndpointsProtocol=https;AccountName={0};AccountKey={1}", _mlstoragename, _mlstoragekey);

            lastScores = GetLatestRetrainedResults(); //Set this for retraining so we can compare updated model score

            //sqlQueryForRetraining = GetSQLQueryFromAzureBlob(); //Set the default SQL Query in case we don't update
            sqlQueryForRetraining = null;
        }

        public bool UploadNewTrainingFileToStorage(string pathToFile)
        {
            if (!File.Exists(pathToFile))
            {
                throw new FileNotFoundException("Path to file not valid");
            }

            Uri u = new Uri(pathToFile);
            string name = u.Segments[u.Segments.Length - 1];

            try {
                using (FileStream fs = File.OpenRead(pathToFile))
                {
                    CloudStorageAccount storageAccount = CloudStorageAccount.Parse(_storageConnectionString);
                    CloudBlobClient blobClient = storageAccount.CreateCloudBlobClient();
                    CloudBlobContainer container = blobClient.GetContainerReference(_mlstoragecontainer);
                    CloudBlockBlob fileBlob = container.GetBlockBlobReference(name);
                    fileBlob.UploadFromStream(fs);

                    _trainingBlob = fileBlob;
                }

                return true;
            }
            catch
            {
                return false;
            }
        }

        /// <summary>
        /// Used to setup the job retraining. You nust receive the jobID from this call and submit to the start method. This method only gets retraining 
        /// ready but it MUST be started with the returned jobID after the sucessful completion of this call
        /// This will submit the default SQL Query pulled from the ML Storage container and use it unless it was nodified with other LIB calls
        /// </summary>
        /// <returns>This is a jobID used to then start the retraining job</returns>
        public async Task<string> QueueRetrainingAsync(TRAINING_DATA_SOURCE source, Dictionary<string,string> globalOptions)
        {
            string jobId = null;
            BatchExecutionRequest request = null;
            AzureBlobDataReference adr = null;

            if (source== TRAINING_DATA_SOURCE.DATA_UPLOAD && _trainingBlob == null)
            {
                throw new Exception("Training set not uploaded referenced");
            }
            else if (source == TRAINING_DATA_SOURCE.DATA_UPLOAD && _trainingBlob !=null)
            {
                adr = new AzureBlobDataReference()
                {
                    ConnectionString = _storageConnectionString,
                    RelativeLocation = _trainingBlob.Uri.LocalPath
                };
            }

            //Now do outputs
            var outputs = new Dictionary<string, AzureBlobDataReference>()
                {
                    {
                        "output2",
                        new AzureBlobDataReference()
                        {
                            ConnectionString = _storageConnectionString,
                            RelativeLocation = string.Format("/{0}/{1}.ilearner", _mlstoragecontainer, _currentModelTraining)
                        }
                    },
                    {
                        "output1",
                        new AzureBlobDataReference()
                        {
                            ConnectionString = _storageConnectionString,
                            RelativeLocation = string.Format("/{0}/{1}.csv", _mlstoragecontainer, _currentModelTraining)
                        }
                    },
                };

            //Now setup request object
            request = new BatchExecutionRequest()
            {
                Outputs = outputs
            };
            if (globalOptions != null)
            {
                request.GlobalParameters = globalOptions;
            }
            else
            {
                request.GlobalParameters = new Dictionary<string, string>();
            }
            if (adr != null)
            {
                request.Input = adr;
            }


            using (HttpClient client = new HttpClient())
            {
                client.DefaultRequestHeaders.Authorization = new AuthenticationHeaderValue("Bearer", _mlretrainerkey);

                // WARNING: The 'await' statement below can result in a deadlock if you are calling this code from the UI thread of an ASP.Net application.
                // One way to address this would be to call ConfigureAwait(false) so that the execution does not attempt to resume on the original context.
                // For instance, replace code such as:
                //      result = await DoSomeTask()
                // with the following:
                //      result = await DoSomeTask().ConfigureAwait(false)

                // submit the job
                string uploadJobURL = _mlretrainmodelurl + "?api-version=2.0";
                HttpResponseMessage response = await client.PostAsJsonAsync(uploadJobURL, request);
                if (!response.IsSuccessStatusCode)
                {
                    return null;
                }
                jobId = await response.Content.ReadAsAsync<string>(); //Used to reference the job for start and monitoring of completion
            }

            return jobId;
        }

        /// <summary>
        /// This is used to check the status of a submitted job for model retraining
        /// </summary>
        /// <param name="jobId"></param>
        /// <returns></returns>
        public async Task<BatchScoreStatusCode> CheckJobStatus(string jobId)
        {
            BatchScoreStatus status;

            using (HttpClient client = new HttpClient())
            {
                // Check the job
                string jobLocation = _mlretrainmodelurl + "/" + jobId + "?api-version=2.0";
                client.DefaultRequestHeaders.Authorization = new AuthenticationHeaderValue("Bearer", _mlretrainerkey);

                HttpResponseMessage response = await client.GetAsync(jobLocation);
                if (!response.IsSuccessStatusCode)
                {
                    return BatchScoreStatusCode.Cancelled;
                }

                status = await response.Content.ReadAsAsync<BatchScoreStatus>();
            }

            return status.StatusCode;
        }

        /// <summary>
        /// Once the job has been queued, this API starts the actual execution of the retraining via a jobID obtained when queueing the job
        /// </summary>
        /// <param name="jobId"></param>
        /// <returns></returns>
        public async Task StartRetrainingJob(string jobId)
        {
            using (HttpClient client = new HttpClient())
            {
                // start the job
                string jobStartURL = _mlretrainmodelurl + "/" + jobId + "/start?api-version=2.0";
                client.DefaultRequestHeaders.Authorization = new AuthenticationHeaderValue("Bearer", _mlretrainerkey);
                HttpResponseMessage response = await client.PostAsync(jobStartURL, null);
                if (!response.IsSuccessStatusCode)
                {
                    return;
                }
            }

        }

        /// <summary>
        /// This method puts in place the retrained iLearner for the published endpoint
        /// </summary>
        /// <param name="baseLoc"></param>
        /// <param name="relLoc"></param>
        /// <param name="sasBlobtoken"></param>
        /// <param name="connStr"></param>
        /// <returns></returns>
        private async Task<bool> UpdateRetrainedModel(string baseLoc, string relLoc, string sasBlobtoken, string connStr)
        {
            var resourceLocations = new ResourceLocations()
            {
                Resources = new ResourceLocation[] {
                    new ResourceLocation()
                    {
                        //Name = "Scenario 1 When will a customer return [trained model]",
                        Name = _nameOfEndpoint,
                        Location = new AzureBlobDataReference()
                        {
                            BaseLocation = baseLoc,
                            RelativeLocation = relLoc,
                            SasBlobToken = sasBlobtoken
                        }
                    }
                }
            };

            using (var client = new HttpClient())
            {
                client.DefaultRequestHeaders.Authorization = new AuthenticationHeaderValue("Bearer", _endpointkey);
                using (HttpRequestMessage request = new HttpRequestMessage(new HttpMethod("PATCH"), _endpointurl))
                {
                    request.Content = new StringContent(JsonConvert.SerializeObject(resourceLocations), System.Text.Encoding.UTF8, "application/json");
                    HttpResponseMessage response = await client.SendAsync(request);
                    if (response.IsSuccessStatusCode) { return true;  }
                    else { return false; }
                }
            }
        }

        private async Task<bool> UpdateRetrainedModel2(string baseLoc, string relLoc, string sasBlobtoken, string connStr)
        {
            var resourceLocations = new ResourceLocations()
            {
                Resources = new ResourceLocation[] {
                    new ResourceLocation()
                    {
                        //Name = "Scenario 1 When will a customer return [trained model]",
                        Name = _nameOfEndpoint,
                        Location = new AzureBlobDataReference()
                        {
                            BaseLocation = baseLoc,
                            RelativeLocation = relLoc,
                            SasBlobToken = sasBlobtoken
                        }
                    }
                }
            };

            using (var client = new HttpClient())
            {
                client.DefaultRequestHeaders.Authorization = new AuthenticationHeaderValue("Bearer", _endpoint2key);
                using (HttpRequestMessage request = new HttpRequestMessage(new HttpMethod("PATCH"), _endpoint2url))
                {
                    request.Content = new StringContent(JsonConvert.SerializeObject(resourceLocations), System.Text.Encoding.UTF8, "application/json");
                    HttpResponseMessage response = await client.SendAsync(request);
                    if (response.IsSuccessStatusCode) { return true; }
                    else { return false; }
                }
            }
        }

        /// <summary>
        /// Used to update the retrained model. It first checks to ensure that the job completed successfuly.
        /// </summary>
        /// <param name="jobId"></param>
        /// <returns></returns>
        public async Task<Boolean> UpdateModel(string jobId)
        {
            BatchScoreStatus status;
            bool isUdpated = false;

            try
            {
                using (HttpClient client = new HttpClient())
                {
                    // Check the job
                    string jobLocation = _mlretrainmodelurl + "/" + jobId + "?api-version=2.0";
                    client.DefaultRequestHeaders.Authorization = new AuthenticationHeaderValue("Bearer", _mlretrainerkey);

                    HttpResponseMessage response = await client.GetAsync(jobLocation);
                    if (!response.IsSuccessStatusCode)
                    {
                        return false;
                    }

                    status = await response.Content.ReadAsAsync<BatchScoreStatus>();
                }

                AzureBlobDataReference res = status.Results["output2"];
                isUdpated = await UpdateRetrainedModel(res.BaseLocation, res.RelativeLocation, res.SasBlobToken, _storageConnectionString);
                //isUdpated = await UpdateRetrainedModel2(res.BaseLocation, res.RelativeLocation, res.SasBlobToken, _storageConnectionString);
                return isUdpated;
            }
            catch
            {
                return false;
            }
        }

        /// <summary>
        /// This method is primarily used to upload a new training set before retraining. 
        /// If you are retraining where the model trainer is fed from a cloud based storage location (SQL or Azure Storage for example) then
        /// you won't need this call
        /// </summary>
        /// <param name="blobName"></param>
        /// <returns></returns>
        public bool GetRetrainingBlob(string blobName)
        {
            string rtBlobName = blobName;

            if (rtBlobName.EndsWith(".csv") || rtBlobName.EndsWith(".nh.csv") || rtBlobName.EndsWith(".tsv") || rtBlobName.EndsWith(".nh.tsv")) { }
            else
            {
                throw new FileNotFoundException(string.Format(CultureInfo.InvariantCulture, "File {0} is not a supported extension type (like csv or tsv)", rtBlobName));
            }

            try
            {
                CloudBlobClient blobClient = CloudStorageAccount.Parse(_storageConnectionString).CreateCloudBlobClient();
                CloudBlobContainer container = blobClient.GetContainerReference(_mlstoragecontainer);
                _trainingBlob = container.GetBlockBlobReference(rtBlobName);

                if (_trainingBlob.Exists())
                {
                    return true;
                }
                else
                {
                    return false;
                }
            }
            catch
            {
                throw new Exception(string.Format(CultureInfo.InvariantCulture, "Blob {0} doesn't exist on local system.", rtBlobName));
            }
        }

        /// <summary>
        /// Used to initially upload a SQL query used for model retraining. We can programmtically send it to the model retrainer and override the default
        /// </summary>
        /// <param name="filePath"></param>
        public void StoreQueryInBlob(string filePath)
        {
            string conn = _storageConnectionString;

            try
            {
                var blobClient = CloudStorageAccount.Parse(conn).CreateCloudBlobClient();
                var container = blobClient.GetContainerReference(_mlstoragecontainer);
                CloudBlockBlob blob = container.GetBlockBlobReference("sqlquery.sql");

                using (var fileStream = System.IO.File.OpenRead(filePath))
                {
                    blob.UploadFromStream(fileStream);
                }
            }
            catch (Exception)
            {

            }
        }
        /// <summary>
        /// This method can be used when kicking off a retrain when the experiment uses an Azure ML Reader supporting SQL. It allows you to modify the 
        /// SQL query string and pass that in as a parameter for retraining. Other methods can support local training file upload, R etc. but this is dedicated to SQL. 
        /// NOTE: There is also a supporting method for this call that does a REGEX date replace in the SQL query before it's sent up to Azure ML for retraining
        /// </summary>
        /// <returns></returns>
        public string GetSQLQueryFromAzureBlob()
        {
            string conn = _storageConnectionString;
            Dictionary<string, Double> vals = null;
            try
            {
                var blobClient = CloudStorageAccount.Parse(conn).CreateCloudBlobClient();
                var container = blobClient.GetContainerReference(_mlstoragecontainer);
                //container.CreateIfNotExists();
                var blob = container.GetBlockBlobReference("sqlquery.sql");
                if (!blob.Exists()) { return null; }
                //blob.DownloadToFile(@"c:\drops\results.csv", FileMode.OpenOrCreate);

                MemoryStream mStream = new MemoryStream();
                blob.DownloadToStream(mStream);

                string decoded = Encoding.UTF8.GetString(mStream.ToArray());;

                return decoded;
            }
            catch
            {
                return null;
            }
        }

        /// <summary>
        /// This method will delete all iLearners (trained modesl) and results files (*.csv) in the specificed Azure Blob Storage Container
        /// If you want to clean out all previous runs of the retrainer, this is how you clean it. It does not remove scracth files or the stored SQL query
        /// </summary>
        public void DeleteAllStoredResultsAndLearners()
        {
            List<CloudBlockBlob> blobList = new List<CloudBlockBlob>();

            string conn = _storageConnectionString;

            try
            {
                var blobClient = CloudStorageAccount.Parse(conn).CreateCloudBlobClient();
                var container = blobClient.GetContainerReference(_mlstoragecontainer);
                //container.CreateIfNotExists();
                var retrainerList = container.ListBlobs(retrainerPrefix, true);
                var res = from b in retrainerList
                          where b.StorageUri.PrimaryUri.AbsoluteUri.EndsWith(".csv") || b.StorageUri.PrimaryUri.AbsoluteUri.EndsWith(".ilearner")
                          select b;
                foreach (CloudBlockBlob blob in res)
                {
                    blobList.Add(blob);
                }

                CloudBlockBlob[] cbArray = blobList.ToArray();
                for (int loop=0; loop<cbArray.Length; loop++)
                {
                    cbArray[loop].Delete();
                }
            }
            catch
            {
                return;
            }

        }

        /// <summary>
        /// This method will return a List of ALL trained model results files order in descending order. Thus you can use this call to help audit past 
        /// model retraining sessions to ensure accuracy improvement
        /// </summary>
        /// <returns></returns>
        public List<Dictionary<string, double>> GetAllStoredResults()
        {
            List<Dictionary<string, double>> resList = new List<Dictionary<string, double>>();

            string conn = _storageConnectionString;

            try
            {
                var blobClient = CloudStorageAccount.Parse(conn).CreateCloudBlobClient();
                var container = blobClient.GetContainerReference(_mlstoragecontainer);
                //container.CreateIfNotExists();
                var retrainerList = container.ListBlobs(retrainerPrefix, true);
                var res = from b in retrainerList
                          where b.StorageUri.PrimaryUri.AbsoluteUri.EndsWith(".csv")
                          orderby b.StorageUri.PrimaryUri.AbsoluteUri descending
                          select b;
                foreach(CloudBlockBlob blob in res)
                {
                    resList.Add(GetTrainingResultsFromBlob(blob));
                }
            }
            catch
            {
                return null;
            }


            return resList;
        }

        /// <summary>
        /// This method retrives the results of the last model retraining. If there are no existing results (it has never run) then it will return null
        /// </summary>
        /// <returns>A filled in Dictionary of model metrics. If none was stored then a single entry with 'nometrics' as the only key</returns>
        public Dictionary<string, double> GetLatestRetrainedResults()
        {
            string conn = _storageConnectionString;
            Dictionary<string, Double> vals = null;
            try
            {
                var blobClient = CloudStorageAccount.Parse(conn).CreateCloudBlobClient();
                var container = blobClient.GetContainerReference(_mlstoragecontainer);
                //container.CreateIfNotExists();
                var retrainerList = container.ListBlobs(retrainerPrefix, true);
                var res = from b in retrainerList
                          where b.StorageUri.PrimaryUri.AbsoluteUri.EndsWith(".csv")
                          orderby b.StorageUri.PrimaryUri.AbsoluteUri descending
                          select b;

                CloudBlockBlob blob = (CloudBlockBlob)res.FirstOrDefault();

                vals = GetTrainingResultsFromBlob(blob);

                return vals;
            }
            catch (Exception)
            {
                return new Dictionary<string, double>(); //Empty
            }
        }

        /// <summary>
        /// This used to get Multiclass results
        /// </summary>
        /// <param name="isString"></param>
        /// <returns></returns>
        public string GetLatestRetrainedResults(bool isString)
        {
            string conn = _storageConnectionString;
            Dictionary<string, Double> vals = null;
            try
            {
                var blobClient = CloudStorageAccount.Parse(conn).CreateCloudBlobClient();
                var container = blobClient.GetContainerReference(_mlstoragecontainer);
                //container.CreateIfNotExists();
                var retrainerList = container.ListBlobs(retrainerPrefix, true);
                var res = from b in retrainerList
                          where b.StorageUri.PrimaryUri.AbsoluteUri.EndsWith(".csv")
                          orderby b.StorageUri.PrimaryUri.AbsoluteUri descending
                          select b;

                CloudBlockBlob blob = (CloudBlockBlob)res.FirstOrDefault();

                return GetTrainingResultsFromBlob(blob, true); ;
            }
            catch (Exception)
            {
                return null; //Empty
            }
        }

        /// <summary>
        /// This method retrives the scoring results from the blob and parses into a Dictionary
        /// </summary>
        /// <param name="blob"></param>
        /// <returns></returns>
        public Dictionary<string, double> GetTrainingResultsFromBlob(CloudBlockBlob blob)
        {
            if (blob == null) //null blob was passed in
            {
                return new Dictionary<string, double>();
            }
            Dictionary<string, double> results = null;

            if (!blob.Exists())
            {
                results = new Dictionary<string, double>();
                results.Add("nometrics", 0);
                return results;
            }

            MemoryStream mStream = new MemoryStream();
            blob.DownloadToStream(mStream);

            string decoded = Encoding.UTF8.GetString(mStream.ToArray());

            results = ExtractModelValues(decoded);

            return results;
        }

        /// <summary>
        /// This method used to get just the string results from the last run
        /// </summary>
        /// <param name="blob"></param>
        /// <param name="isString"></param>
        /// <returns></returns>
        public string GetTrainingResultsFromBlob(CloudBlockBlob blob, bool isString)
        {
            if (blob == null) //null blob was passed in
            {
                return null;
            }

            if (!blob.Exists())
            {
                return null;
            }

            MemoryStream mStream = new MemoryStream();
            blob.DownloadToStream(mStream);

            string decoded = Encoding.UTF8.GetString(mStream.ToArray());

            return decoded;
        }

        /// <summary>
        /// This takes the downloaded file contents and parses them into a dictionary of scoring values. There are several so the implementer needs to decide which
        /// ones they want to use for determining if the retrained model should be deployed or not. A default is usually AUC
        /// </summary>
        /// <param name="mvalues"></param>
        /// <returns></returns>
        private Dictionary<string, Double> ExtractModelValues(string mvalues)
        {
            Dictionary<string, Double> vals = new Dictionary<string, double>();

            string[] entries = mvalues.Split(new char[] { '\r', '\n' }, StringSplitOptions.RemoveEmptyEntries);

            string[] headers = entries[0].Split(',');
            string[] values = entries[1].Split(',');

            for (int loop = 0; loop < headers.Length; loop++)
            {
                vals.Add(headers[loop], Convert.ToDouble(values[loop]));
            }

            return vals;
        }

        /// <summary>
        /// This method will tell you if you should deploy the retrained model. it compares the last known scores with the newest scores
        /// from retraining. Note that when this class is instantiated, it will attempt to retrieve last results from the configured blob. 
        /// Thus if there are no existing scores in the blob, it will return TRUE since it assumes no deployed endpoint yet.
        /// </summary>
        /// <param name="measuredValue"></param>
        /// <param name="pctImproveMin">This should be in it's decimal form such as 0.02f for 2% improvement target as an exmaple</param>
        /// <returns></returns>
        public bool isUdpateModel(string measuredValue, float pctImproveMin)
        {
            if (String.IsNullOrEmpty(measuredValue) || pctImproveMin == 0 || retrainedScores == null)
            {
                return false;
            }
            if (lastScores == null) { return true;
            }
            bool isImproved = false;

            var lastCompareScore = lastScores[measuredValue];
            var newCompareScore = retrainedScores[measuredValue];

            var improvement = lastCompareScore + (lastCompareScore * pctImproveMin);
            if (newCompareScore >= improvement) { isImproved = true; }

            return isImproved;
        }

        public void UpdateSQLQueryForNewDate(string newDate)
        {
            Regex rgx = new Regex(@"\d{4}\-\d{2}-\d{2}");
            if (!rgx.IsMatch(newDate))
            {
                return; //Not a valid date
            }
            string result = rgx.Replace(sqlQueryForRetraining, newDate);
            sqlQueryForRetraining = result;
        }

        /// <summary>
        /// This class is used to send the config array to the constructor of the ML Retrainer Library
        /// </summary>
        public class MLRetrainConfig
        {
            public string mlretrainerurl { get; set; }
            public string mlretrainerkey { get; set; }
            public string publishendpointurl { get; set; }
            public string publishendpointkey { get; set; }
            public string publishendpointname { get; set; }
            public string mlretrainerstoragename { get; set; }
            public string mlretrainerstoragekey { get; set; }
            public string mlretrainercontainername { get; set; }
            public string publishendpoint2url { get; set; }
            public string publishendpoint2key { get; set; }

        }
    }

    public class AzureBlobDataReference
    {
        // Storage connection string used for regular blobs. It has the following format:
        // DefaultEndpointsProtocol=https;AccountName=ACCOUNT_NAME;AccountKey=ACCOUNT_KEY
        // It's not used for shared access signature blobs.
        public string ConnectionString { get; set; }

        // Relative uri for the blob, used for regular blobs as well as shared access 
        // signature blobs.
        public string RelativeLocation { get; set; }

        // Base url, only used for shared access signature blobs.
        public string BaseLocation { get; set; }

        // Shared access signature, only used for shared access signature blobs.
        public string SasBlobToken { get; set; }
    }

    public class ResourceLocations
    {
        public ResourceLocation[] Resources { get; set; }
    }

    public class ResourceLocation
    {
        public string Name { get; set; }
        public AzureBlobDataReference Location { get; set; }
    }

    public class BatchExecutionRequest
    {
        public AzureBlobDataReference Input { get; set; }
        public IDictionary<string, string> GlobalParameters { get; set; }

        // Locations for the potential multiple batch scoring outputs
        public IDictionary<string, AzureBlobDataReference> Outputs { get; set; }
    }

    public enum BatchScoreStatusCode
    {
        NotStarted,
        Running,
        Failed,
        Cancelled,
        Finished
    }

    public class BatchScoreStatus
    {
        // Status code for the batch scoring job
        public BatchScoreStatusCode StatusCode { get; set; }


        // Locations for the potential multiple batch scoring outputs
        public IDictionary<string, AzureBlobDataReference> Results { get; set; }

        // Error details, if any
        public string Details { get; set; }
    }
}
