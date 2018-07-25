using System.Collections;
using System.Collections.Generic;
using System.Net;
using System.Net.Sockets;
using System.Text;
using UnityEngine;
using System.Threading;

public class ArmBrain : MonoBehaviour {

    Thread mThread;
    public string connectionIP = "127.0.0.1";
    public int connectionPort = 25001;
    IPAddress localAdd;
    TcpListener listener;
    TcpClient client;
    Vector3 moveForce = new Vector3();
    Rigidbody rb;
    Rigidbody camBod;
    Vector3 ofsetVec = new Vector3(15, 20, 0);
    bool running;
    public Debugger Debug;
    public float delay = 1;
    public int xStart = 180;
    private void Update()
    {
        if (-81.25 < rb.position.z && rb.position.z < 81.25)
        {
            Move();
        }
        else if (rb.position.z <= -81.25 && moveForce.z > 0)
        {
            Move();
        }
        else if (rb.position.z >= 81.25 && moveForce.z < 0)
        {
            Move();
        }
    }
    IEnumerator ReturnToSender()
    {
        yield return new WaitForSeconds(delay);
        rb.MovePosition(new Vector3(xStart,rb.position.y, rb.position.z));
        camBod.MovePosition(rb.position + ofsetVec);
    }
    private void Move()
    {
        if (moveForce.x > 0)
            StartCoroutine(ReturnToSender());
        rb.AddForce(moveForce,ForceMode.VelocityChange);
        camBod.MovePosition(rb.position + ofsetVec);
        moveForce = moveForce * 0;//reset vector after each move
                                  //check position isnt wall
    }
    private void Start()
    {
        rb = GetComponent<Rigidbody>();
        GameObject view = GameObject.FindGameObjectsWithTag("ArmView")[0];
        camBod = view.GetComponent<Rigidbody>();
        ThreadStart ts = new ThreadStart(GetInfo);
        mThread = new Thread(ts);
        mThread.Start();
        
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

    void GetInfo()
    {
        localAdd = IPAddress.Parse(connectionIP);
        listener = new TcpListener(IPAddress.Any, connectionPort);
        listener.Start();

        client = listener.AcceptTcpClient();
        Debug.Log("got client");
        running = true;
        while (running)
        {
            Connection();
        }
        listener.Stop();
    }

    void Connection()
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
            else
            {
                moveForce = StringToVector3(dataReceived);

                nwStream.Write(buffer, 0, bytesRead);
            }
        }
    }

    public static Vector3 StringToVector3(string sVector)
    {
        // Remove the parentheses
        if (sVector.StartsWith("(") && sVector.EndsWith(")"))
        {
            sVector = sVector.Substring(1, sVector.Length - 2);
        }

        // split the items
        string[] sArray = sVector.Split(',');

        // store as a Vector3
        Vector3 result = new Vector3(
            float.Parse(sArray[0]),
            float.Parse(sArray[1]),
            float.Parse(sArray[2]));

        return result;
    }
  }

