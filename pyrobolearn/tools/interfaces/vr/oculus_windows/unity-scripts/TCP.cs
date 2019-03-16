using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Collections;
using System;
using System.IO;
using System.Net.Sockets;


public class TCP : MonoBehaviour {

    internal Boolean socketReady = false;
    TcpClient mySocket;
    NetworkStream theStream;
    StreamWriter theWriter;
    StreamReader theReader;
    //String Host = "localhost";
    //Int32 Port = 5111;
    void Start()
    {
    }
    void Update()
    {
    }
    // **********************************************
    public void setupSocket(String Host, Int32 Port)
    {
        try
        {
            mySocket = new TcpClient(Host, Port);
            theStream = mySocket.GetStream();
            theWriter = new StreamWriter(theStream);
            theReader = new StreamReader(theStream);
            socketReady = true;
        }
        catch (Exception e)
        {
            Debug.Log("Socket error: " + e);
        }
    }
    public void writeSocket(string theLine)
    {

        if (!socketReady)
            return;
        //Debug.Log("socket:" + theLine);
        String foo = theLine + "\r\n";
        theWriter.Write(foo);
        theWriter.Flush();
    }
    public String readSocket()
    {
        if (!socketReady)
            return "";
        //Debug.Log("reader");
        if (theStream.DataAvailable)
            return theReader.ReadLine();
        //Debug.Log("reader2");
        return "";
    }
    public void closeSocket()
    {
        if (!socketReady)
            return;
        theWriter.Close();
        theReader.Close();
        mySocket.Close();
        socketReady = false;
    }

}
