using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class PlayMovieTextureOnUI : MonoBehaviour {

    public RawImage rawimage;
    public bool createWebcam;
    Texture2D texture;
    TCPImage tcpImage;

    // Use this for initialization
    void Start () {
        createWebcam = false;
        if (createWebcam)
        {
            Debug.Log("Create Webcam");
            WebCamTexture webcamTexture = new WebCamTexture();
            //rawimage.texture = webcamTexture;
            rawimage.material.mainTexture = webcamTexture;
            Debug.Log("Play Webcam");
            webcamTexture.Play();
            Debug.Log("Play Webcam");
        }

        //rawimage.material.mainTexture = texture;

        String host = "10.255.24.97"; //"10.255.24.140";
        Int32 port = 5112;
        tcpImage = new TCPImage();
        //tcpImage.setupSocket(Host, Port);
        //tcpImage.setTexture(rawimage.material.mainTexture);
        //tcpImage.setRawImage(rawimage);
        tcpImage.init(host, port, rawimage);
    }
	
	// Update is called once per frame
	void Update () {
        //tcpImage.readThread();
    }
}
