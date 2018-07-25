using System.Collections;
using System.Collections.Generic;
using System.Net;
using UnityEngine;
using System.Net.Sockets;
using System.Threading;
using System.Text;
using UnityEngine.UI;
public class SocketHandler : MonoBehaviour {
    public static SocketHandler Instance;
    public string connectionIP = "127.0.0.1";
    public int connPort = 25000;
    public Control control ;
    IPAddress localAdd;
    TcpListener listener;
    TcpClient client;
    IPAddress CrashedlocalAdd;
    TcpListener Crashedlistener;
    TcpClient Crashedclient;
    bool running = true;
    bool returned = false;
    bool paused = false;
    // Use this for initialization
    private void Start()
    {
        ThreadStart ts = new ThreadStart(getInfo);//start socket listening thread
        Thread mThread = new Thread(ts);
        mThread.Start();
        ThreadStart cts = new ThreadStart(CrahedgetInfo);//start socket listening thread
        Thread cmThread = new Thread(cts);
        cmThread.Start();
    }
    void getInfo()
    {
        localAdd = IPAddress.Parse(connectionIP);
        listener = new TcpListener(IPAddress.Any, connPort);
        listener.Start();

        client = listener.AcceptTcpClient();
        Debug.Log("got client");
        running = true;
        while (running)
        {
            connection();
        }
        listener.Stop();
    }

    public void TimeChange(float value)
    {
        Time.timeScale = value;
    }
    public void OnApplicationQuit()
    {
        running = false;
    }
    void connection()
    {
        NetworkStream nwStream = client.GetStream();
        byte[] buffer = new byte[client.ReceiveBufferSize];

        int bytesRead = nwStream.Read(buffer, 0, client.ReceiveBufferSize);
        string dataReceived = Encoding.UTF8.GetString(buffer, 0, bytesRead);

        if (dataReceived != null)
        {
            if (dataReceived == "stop")
            {
                running = false;
            }
            else//could get a poll for the rpy or get a set command
            {
                if (dataReceived == "GET ROTATION")
                {
                    string returnString = control.localrotate.x + "," + control.localrotate.y + "," + control.localrotate.z;
                    byte[] data = Encoding.UTF8.GetBytes(returnString.ToCharArray());
                    nwStream.Write(data, 0, data.Length);
                }
                else if (dataReceived == "GET VELOCITY")
                {
                    string returnString = control.localVelocity.x + "," + control.localVelocity.y + "," + control.localVelocity.z;
                    byte[] data = Encoding.UTF8.GetBytes(returnString.ToCharArray());
                    nwStream.Write(data, 0, data.Length);
                }
                else if (dataReceived == "RESET")
                {
                    nwStream.Write(Encoding.UTF8.GetBytes("RESET"), 0, "RESET".Length);
                    control.resetBool = true;
                }
                else
                {
                    control.setThrusts(dataReceived);
                    nwStream.Write(buffer, 0, bytesRead);
                }

            }
        }
    }
    void CrahedgetInfo()
    {
        CrashedlocalAdd= IPAddress.Parse(connectionIP);
        Crashedlistener= new TcpListener(IPAddress.Any, connPort + 1);
        Crashedlistener.Start();

        Crashedclient = Crashedlistener.AcceptTcpClient();
        Debug.Log("got client 2");
        running = true;
        while (running)
        {
            Crashedconnection();
        }
        Crashedlistener.Stop();
    }
    void Crashedconnection()
    {
        NetworkStream nwStream = Crashedclient.GetStream();
        if (control.crashed)
        {
            Debug.Log("CRASHED");
            nwStream.Write(Encoding.UTF8.GetBytes("CRASHED"),0,"CRASHED".Length);
            returned = true;
            control.crashed = false;
        }

    }
    public static string GetLocalIPAddress()
    {
        var host = Dns.GetHostEntry(Dns.GetHostName());
        foreach (var ip in host.AddressList)
        {
            if (ip.AddressFamily == AddressFamily.InterNetwork)
            {
                return ip.ToString();
            }
        }
        throw new System.Exception("No network adapters with an IPv4 address in the system!");
    }
    // Update is called once per frame
    void Update () {
		
	}
}