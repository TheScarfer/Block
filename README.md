# Block
## ⚠️ API Limitations
This demo uses a free API key with limited requests. 
If AI stops responding during testing, the key may have reached its limit.

For full functionality, get your own API key from:
- OpenRouter (free tier): https://openrouter.com/
- Or OpenAI: https://platform.openai.com/

Add to `block.env`:

## 🎯 **Option 2: Add Error Handling** (Better)
Modify your chat function to handle API limits gracefully:

```python
except Exception as e:
    if "rate limit" in str(e).lower():
        return ChatResponse(
            response="I've reached my free API limit for now. For continuous access, please add your own API key to block.env",
            success=False,
            error="API rate limit reached"
        )
    else:
        return ChatResponse(
            response="I'm having technical difficulties. Please try again later.",
            success=False,
            error=str(e)
        )
