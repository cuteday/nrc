
struct Sample
{
    uint val;
};

RWStructuredBuffer<Sample> gSample;

void recordSample(uint val)
{
    uint i = gSample.IncrementCounter();
    if (i < 10000)
    {
        Sample sample = { val };
        gSample[i] = sample;
    }
}
