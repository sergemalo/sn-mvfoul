from SoccerNet.Downloader import SoccerNetDownloader as SNdl

mySNdl = SNdl(LocalDirectory="dataset")
mySNdl.downloadDataTask(task="mvfouls", split=["train","valid","test","challenge"], version="224p", password="s0cc3rn3t")
 