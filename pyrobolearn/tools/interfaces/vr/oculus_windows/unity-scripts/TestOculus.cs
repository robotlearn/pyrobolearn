using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;

public class TestOculus : MonoBehaviour {
    TCP tcp = new TCP();
    //UDP udp;
    OVRCameraRig camera_rig;
    OVRPlayerController player_controller;
    OVRManager manager;
    public bool enableLog = false;
    public bool connectTCP = false;
    //public bool connectUDP = false;
    public AudioClip clip;
    private List<string> dataList;
    private int total;

    OculusHaptics haptics;

    //TCPImage tcpImage;

    // Use this for initialization
    void Start () {
        connectTCP = true;
        //connectUDP = false;
        dataList = new List<string>();

        log("->Start()");

        String Host = "10.255.24.97"; //"10.255.24.140";
        Int32 Port = 5113;
        if (connectTCP)
            tcp.setupSocket(Host, Port);

        //if (connectUDP)
        //{
        //    Debug.Log("Setting UDP");
        //    udp = new UDP();
        //    udp.setupUDP(Port);
        //    Debug.Log("UDP set");
        //}

        //tcpImage = new TCPImage();
        //tcpImage.setupSocket(Host, Port);


        camera_rig = GameObject.FindObjectOfType<OVRCameraRig>();
        player_controller = GameObject.FindObjectOfType<OVRPlayerController>();
        manager = GameObject.FindObjectOfType<OVRManager>();

        log("Start()->");

        // haptics
        haptics = new OculusHaptics();
    }
	





    // Update is called once per frame
    void Update()
    {
        OVRInput.Update();
        log("->Update()!");

        Transform root_anchor = camera_rig.trackingSpace;
        Transform centerEyeAnchor = camera_rig.centerEyeAnchor;
        Transform leftHandAnchor = camera_rig.leftHandAnchor;
        Transform rightHandAnchor = camera_rig.rightHandAnchor;


        Vector3 eye_P_right_hand = rightHandAnchor.position; // - center_eye_anchor.position;

        Quaternion rightHandQuat = rightHandAnchor.rotation;
        Quaternion leftHandQuat = leftHandAnchor.rotation;
        Quaternion headQuat = centerEyeAnchor.rotation;

        string msg;

        msg = string.Format("{0:N5}", -headQuat[2]) + ',' + string.Format("{0:N5}", headQuat[0]) + ',' + string.Format("{0:N5}", -headQuat[1]) + ',' + string.Format("{0:N5}", headQuat[3]);
        //msg = string.Format("{0:N5}", eye_P_right_hand[2]) + ',' + string.Format("{0:N5}", -eye_P_right_hand[0]) + ',' + string.Format("{0:N5}", eye_P_right_hand[1]) + ',';
        //msg += string.Format("{0:N5}", rightHandQuat[2]) + ',' + string.Format("{0:N5}", rightHandQuat[0]) + ',' + string.Format("{0:N5}", rightHandQuat[1]) + ',' + string.Format("{0:N5}", rightHandQuat[3]);
        //msg += string.Format("{0:N5}", -right_hand_quat[2]) + ',' + string.Format("{0:N5}", right_hand_quat[3]) + ',' + string.Format("{0:N5}", right_hand_quat[1]) + ',' + string.Format("{0:N5}", right_hand_quat[0]);


        log(msg);

        if (connectTCP)
        {
            //Debug.Log("tcp");
            //Debug.Log(msg);
            tcp.writeSocket(msg);
            //String str = tcp.readSocket();
            //Debug.Log(str);
            //Debug.Log(str.Length);
            //if (str.Equals("end")) {
            //    Debug.Log("End");
            //    string data = string.Join("", dataList.ToArray());
            //    dataList.Clear();
            //    total = 0;
            //    Debug.Log("Total length: ");
            //    Debug.Log(data.Length);
            //} else {
            //    if (str.Length != 0) {
            //        total += str.Length;
            //        Debug.Log(total);
            //        dataList.Add(str);
            //    }
            //}
        }

        //String s = tcpImage.readThread();
        //if (s.Length != 0) {
        //    Debug.Log("Length of image received: " + s.Length);
        //    //Debug.Log("First letters: " + s.Substring(0, 20));
        //}
        //if (connectUDP) {
        //    string data = udp.readSocket();
        //    Debug.Log(data.Length);
        //}

        // haptics
        // WARNING: To get it working, you have to wear the headset or activate its proximity sensor!!
        if (OVRInput.Get(OVRInput.Button.One))
        {

            Debug.Log("Button A pressed!");
            haptics.Vibrate(VibrationForce.Light);
        }

        if (OVRInput.Get(OVRInput.Button.Two))
        {

            Debug.Log("Button B pressed!");
            haptics.Vibrate(VibrationForce.Medium);
        }

        if (OVRInput.Get(OVRInput.Button.Three))
        {

            Debug.Log("Button X pressed!");
            haptics.Vibrate(VibrationForce.Hard);
        }

        log("  Update()->");
    }

    private void log(string msg) {
        if (enableLog)
            Debug.Log(msg);
    }

    private void logWarning(string msg) {
        if (enableLog)
            Debug.LogWarning(msg);
    }




}

