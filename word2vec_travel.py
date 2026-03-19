import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from gensim.models import Word2Vec

#  Download NLTK Dependencies
import nltk
nltk.download('punkt')       # Tokenization tool
nltk.download('stopwords')   # Stop words corpus

#
text = """
Daily Life, Nature and Travel
In the hustle and bustle of modern urban life, many people forget to pause and appreciate the beauty of nature that surrounds them—yet travel offers a perfect chance to reconnect with the natural world beyond the city limits. Every morning, as the sun rises over the city skyline, golden light filters through the leaves of old oak trees in the neighborhood park, but for those who love to travel, the sunrise over a mountain lake or a coastal beach is an even more precious sight. Families walk their dogs along winding paths in local parks, but on weekends and holidays, they often travel to nearby towns, national parks, or rural villages to escape the noise of the city and breathe fresh air.

Travel is not just about visiting new places; it is about exploration, adventure, and learning. A short trip to a countryside village can teach you about local traditions, while a longer journey to a foreign country opens your eyes to different cultures, languages, and ways of life. When you travel through mountain ranges, you hike along trails lined with pine trees and wild berries, listen to the sound of mountain streams, and watch eagles soar above snow-capped peaks. Travel to coastal regions lets you walk along sandy beaches, collect seashells, and taste fresh seafood caught by local fishermen. 

Many people find joy in planning their travel: researching destinations, booking accommodation, packing a backpack with essentials, and creating a list of places to visit—from historic castles and museums to hidden waterfalls and quiet forests. Even a simple day trip to a nearby lake can feel like an adventure, as you row a boat across calm water, fish for trout, or have a picnic with family and friends. Travel also teaches patience: delayed trains, unexpected weather, or language barriers are small challenges that make the journey more memorable.

Nature is the greatest companion for travel. When you travel to a national park, you encounter deer grazing in meadows, hear the call of woodpeckers in forests, and smell the sweet scent of pine and cedar in the air. In spring, travel to cherry blossom groves in Japan or tulip fields in the Netherlands, and in autumn, travel to New England to see maple leaves turn fiery red and orange, crunching underfoot as you walk through quiet woods. 

Reading is another way to fuel your desire to travel—books about travel memoirs, adventure novels, or guidebooks transport you to far-off lands, from the streets of Paris to the mountains of Nepal, even when you cannot leave your home. A good travel book, like a real trip, can make time slow down, allowing your mind to wander and dream of future journeys. 

Food is an essential part of travel too. When you travel, you taste local dishes: fresh pasta in Italy, spicy tacos in Mexico, or steaming bowls of ramen in Japan. Farmers’ markets in the cities you travel to offer juicy strawberries, crisp lettuce, and ripe tomatoes, while street vendors sell warm bread, savory pastries, and sweet treats that reflect the local culture. Sharing a meal with strangers you meet while traveling creates connections that last a lifetime.

As the day ends and the sun sets, whether you are at home or traveling in a foreign land, the sky painted in hues of purple and orange brings a sense of peace. Travel reminds us that happiness is found not just in routine daily life, but in the small moments of adventure: a bird’s song in a foreign forest, a breeze through an open window of a train as you travel across the countryside, or the taste of a fresh mango from a market in Thailand. Taking the time to travel, explore, and connect with nature makes even the most ordinary days feel rich and meaningful.
"""

# Text preprocessing function
def preprocess_text(text):
    # 1. Convert to lowercase
    text = text.lower()
    # 2. Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # 3. Tokenization
    tokens = word_tokenize(text)
    # 4. Filter stop words and empty characters
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token not in stop_words and token.strip()]
    # 5. Group by sentences
    sentences = [filtered_tokens]
    return sentences

# Execute preprocessing
processed_sentences = preprocess_text(text)
print("Number of words after preprocessing:", len(processed_sentences[0]))
print("Preprocessing example (first 20 words):", processed_sentences[0][:20])

#  Train Word2Vec Model
model = Word2Vec(
    sentences=processed_sentences,
    vector_size=100,
    window=5,
    min_count=1,
    sg=1,
    negative=5,
    epochs=200
)

# Query Words Similar to 'travel'
print("\n" + "="*60)

# Core modification: Query words similar to 'travel'
print("1. Words similar to 'travel':")
similar_travel = model.wv.most_similar("travel", topn=10)
for word, similarity in similar_travel:
    print(f"   {word} → Similarity: {similarity:.4f}")

# View word vector of 'travel'
print("\n Word vector of 'travel' (first 10 dimensions):")
print(model.wv["travel"][:10])