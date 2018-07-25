using UnityEngine;

public class terrainGenerator : MonoBehaviour
{
    public int maxHeight = 20;
    public int scale = 20;
    public int width = 500;
    public int height = 500;
    public GameObject target;
    public GameObject cam;
    private void Update()
    {
        GameObject target = GameObject.FindGameObjectWithTag("TARGET");
        transform.position = new Vector3(target.transform.position.x-250, 0, target.transform.position.z-250);
        cam.transform.position = new Vector3(transform.position.x + 250 , 500, transform.position.z+ 250);
        Terrain terrain = GetComponent<Terrain>();
        terrain.terrainData = GenTerrain(terrain.terrainData);
        
    }
    private TerrainData GenTerrain(TerrainData terrainData)
    {
        terrainData.heightmapResolution = width + 1;
        terrainData.size = new Vector3(width, maxHeight, height);
        terrainData.SetHeights(0, 0, GenerateHeights());
        return terrainData;
    }
    private float[,] GenerateHeights()
    {
        float[,] heights = new float[width, height];
        for (int x = 0; x < width; x++)
        {
            for (int y = 0; y < height; y++)
            {
                heights[x, y] = getHeight(x,y);
            }
        }
        return heights;
    }
    float getHeight(int x, int y)
    {
        float xC = (float) x / width * scale;
        float yC = (float) y / height * scale;
        return Mathf.PerlinNoise(xC, yC);
    }
}
