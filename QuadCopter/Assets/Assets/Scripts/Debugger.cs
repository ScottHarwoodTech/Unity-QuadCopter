using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class Debugger : MonoBehaviour {

    // Use this for initialization
    public Text text;
    public int lineNums = 10;
    public string storedText;
    private List<string> lines = new List<string>();
	void Start () {
		
	}
	
	// Update is called once per frame
	void Update () {
        storedText = "";
        for (int line = 0; line<lines.Count; line++)
        {
            storedText = storedText + lines[line];
        }
        text.text = storedText;
	}
    public void Log(string text)
    {
        lines.Add(text + "\n") ;
    }
}
