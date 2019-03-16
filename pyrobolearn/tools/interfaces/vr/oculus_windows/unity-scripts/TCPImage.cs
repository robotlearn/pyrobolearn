using System;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using UnityEngine;
using UnityEngine.UI;

public class TCPImage : MonoBehaviour {
    // mainly inspired by: https://stackoverflow.com/questions/42717713/unity-live-video-streaming

    // std variables
    public bool enableLog = false;
    private bool stop = false;
    //private bool displayImage = false;

    // network variables
    private TcpClient client;
    private String host;
    private Int32 port;
    private bool socketReady = false;
    NetworkStream stream;

    // Image variables
    // nb of bytes used to specify the length of the image (needs to be the same on the server side)
    private const int IMAGE_LENGTH = 4; // 4 bytes are enough (=int32)
    Texture2D imageTexture;
    RawImage rawImage;

    // Thread
    Thread thread;
    static readonly object lockObj = new object();
    byte[] imageBytes;


    // Use this for initialization
    void Start () {
	}
	
	// Update is called once per frame
	void Update () {
        // display image
        //if (displayImage) {
        //    bool loaded = false;
        //    lock (lockObj) {
        //        loaded = imageTexture.LoadImage(imageBytes);
        //        log("Image loaded: " + loaded);
        //    }
        //    rawImage.texture = imageTexture;
        //    displayImage = false;
        //    if (loaded)
        //        Debug.Log("Image loaded!");
        //}
	}

    public void init(String host, Int32 port, RawImage image) {
        // set variables
        this.imageTexture = new Texture2D(0, 0);
        this.rawImage = image;
        this.host = host;
        this.port = port;

        // create new thread that will update rawimage with the new incoming images
        //thread = new Thread(readThread);
        //thread.IsBackground = true;
        //thread.Start();
        Loom.RunAsync(() => {
            readThread();
        });
    }

    public void readThread() {
        // create client (connect to server)
        Debug.Log("Setup TCP Image connection");
        this.client = new TcpClient(this.host, this.port);

        // thread main loop: read images from the network
        while (!this.stop) {
            // read image size
            int dim = readImageSize(IMAGE_LENGTH);
            log("Dimension of image to be received: " + dim);

            // read image
            if (dim > 0) {
                readImage(dim);
            } else {
                logWarning("Lost Connection!!!");
            }
        }
    }

    /**Read the specified number of bytes from the network.
     * param: bytes - this will be filled by what is received from the network.
     */
    private bool read(byte[] bytes) {
        stream = this.client.GetStream();
        int total = 0;
        int size = bytes.Length;
        while (total != size) {
            int nbBytesRead = stream.Read(bytes, total, size - total);
            if (nbBytesRead == 0) // disconnected
                return false;
            total += nbBytesRead;
        }
        return true;
    }

    /**Read the size of the image that will sent through the network.
     */
    private int readImageSize(int size) {
        byte[] imageSizeBytes = new byte[size];
        bool connected = read(imageSizeBytes);
        if (!connected)
            return -1;
        int imageSize = BitConverter.ToInt32(imageSizeBytes, 0);
        return imageSize;
    }

    /**Read the image from the network using the specified size.
     */
    //private bool readImage(int size) {
     private void readImage(int size) {
        // read the image from the network
        //byte[] imageBytes = new byte[size];
        bool connected = false;
        //lock (lockObj) {
            imageBytes = new byte[size];
            connected = read(imageBytes);
            log("image loaded in bytes!");
        //displayImage = true;
        //}

        bool ready = false;

        // displayImage
        if (connected) {
            Loom.QueueOnMainThread(() => {
                displayImage(imageBytes);
                ready = true;
            });
        }

        // wait until old Image is displayed
        while (!ready)
            System.Threading.Thread.Sleep(1);

        //return connected;
    }

    private void displayImage(byte[] imageBytes) {
        bool loaded = imageTexture.LoadImage(imageBytes);
        log("Image loaded: " + loaded);
        this.rawImage.texture = imageTexture;
    }

    private void log(string msg) {
        if (enableLog)
            Debug.Log(msg);
    }

    private void logWarning(string msg) {
        if (enableLog)
            Debug.LogWarning(msg);
    }

    private void OnApplicationQuit() {
        logWarning("Quitting application!");
        this.stop = true;
        if (this.client != null)
            this.client.Close();
    }






    public void setupSocket(String Host, Int32 Port) {
        imageTexture = new Texture2D(0, 0);
        try {
            Debug.Log("Setup TCP IMAGE connection");
            client = new TcpClient(Host, Port);
            //buffer = new byte[client.ReceiveBufferSize];
            //memStream = new MemoryStream();
            //stream = client.GetStream();
            socketReady = true;
        }
        catch (Exception e)
        {
            Debug.Log("Socket error: " + e);
        }
        if (socketReady) {
            
        }
    }

    public void setRawImage(RawImage image) {
        rawImage = image;
    }

    //public String readThread() {
    //    if(socketReady) {
    //        //while(true) { // keep reading
    //        //Debug.Log("Collecting data...");
    //        memStream = new MemoryStream();
    //        while (true) { //(stream.DataAvailable) { // read an image
    //            int len = stream.Read(buffer, 0, buffer.Length);
    //            if (len <= 0)
    //                break;
    //            //Debug.Log(len);
    //            //Debug.Log(stream.DataAvailable);
    //            count += 1;
    //            Debug.Log(count);
    //            memStream.Write(buffer, 0, len);
    //        }
    //        count = 0;
    //        //Debug.Log("Data collected!");
    //        // memStream.Seek(0, SeekOrigin.Begin);
    //        //StreamReader reader = new StreamReader(memStream);
    //        //string str = reader.ReadToEnd(); //Encoding.ASCII.GetString(memStream.ToArray());
    //        string str = Encoding.ASCII.GetString(memStream.ToArray());
    //        //Debug.Log(str.Length);
    //        //memStream.SetLength(0);
    //        return str;
    //    } else {
    //        Debug.log("socket not ready");
    //        return "";
    //    }
    //}

    //public void readThread() {
    //    if (socketReady) {
    //        int dim = readImageSize(IMAGE_LENGTH);
    //        Debug.Log("Dimension of image to be received: " + dim);
    //        string str = "";
    //        if (dim > 0) {
    //            str = readImage(dim);
    //            Debug.Log("Length of image received: " + str.Length);
    //        } else
    //            Debug.Log("Lost Connection!!!");
    //    } else {
    //        Debug.Log("socket not ready!");
    //    }
    //}

}
