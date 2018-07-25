using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class PuckBrain : MonoBehaviour {
    public Rigidbody rb;
    public Text PuckSpeed;
    // Use this for initialization
    public Debugger Debug;
    void Start () {
        Debug.Log("I am alive!");
        rb = GetComponent<Rigidbody>();
        rb.AddForce(500f, 0, Random.Range(20, 40), ForceMode.Impulse);

    }
	
	// Update is called once per frame
	void Update () {
        PuckSpeed.text = "Puck Speed: \n " + rb.velocity.ToString();
    }
    public void Reset()
    {
        Debug.Log("RESET");
        rb.MovePosition(new Vector3(0, 2.5f, 0));
        rb.velocity = Vector3.zero;
        rb.AddForce(500f, 0, Random.Range(20, 40), ForceMode.Impulse);

    }
}
