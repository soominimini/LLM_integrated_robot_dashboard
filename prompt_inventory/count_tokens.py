import sys, types, os
# --- load src/.env without printing secrets ---
for raw in open('src/.env'):
    line=raw.strip()
    if not line or line.startswith('#'): continue
    if line.startswith('export '): line=line[7:]
    if '=' not in line: continue
    k,v=line.split('=',1); v=v.strip().strip('"').strip("'")
    os.environ.setdefault(k,v)

sys.path.insert(0,'src')
from story_generator import StoryGenerator
from knowledge_base import LanguageInterestKB
import anthropic
from google import genai
from google.genai import types

an=anthropic.Anthropic()  # ANTHROPIC_API_KEY
gem=genai.Client(api_key=os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY'))

CTX={'claude-sonnet-4-6':1_000_000,'gemini-2.5-flash':1_048_576,
     'gemini-2.5-flash-image':32_768,'gemini-robotics-er-1.6-preview':131_072}
IMG='image_dataset/grape.jpg'; img_bytes=open(IMG,'rb').read()

def claude_ct(system,user,model='claude-sonnet-4-6'):
    return an.messages.count_tokens(model=model,system=system,
        messages=[{"role":"user","content":user}]).input_tokens
def gem_ct(text,model='gemini-2.5-flash',image=False):
    contents=[types.Part.from_bytes(data=img_bytes,mime_type='image/jpeg'),text] if image else [text]
    return gem.models.count_tokens(model=model,contents=contents).total_tokens

SYS_STORY=("You are a clinical storyteller for pediatric speech-language therapy. You create "
"personalized therapeutic stories that are age-calibrated, clinically grounded, and engaging. "
"You follow word count constraints precisely. You integrate therapy goals into narrative and "
"dialogue naturally, never as explicit lessons. You never add preamble, commentary, or text "
"outside the requested output format.")
SAR=open('documents/sar_system_prompt.md').read()
kb=LanguageInterestKB(); sg=StoryGenerator(llm_model='claude-sonnet-4-6')

print("=== STORY main prompt — EXACT (Anthropic count_tokens), system+user ===")
for age in (3,4,5,6,7,8,9):
    persona=kb.build_story_prompt_fragment(age,'boy')
    user=sg._build_prompt('Alex',age,'boy',['friends'],'',persona)
    t=claude_ct(SYS_STORY,user)
    print(f" age {age:>2}: {t:>5} tok  ({100*t/1_000_000:.3f}% of 1,000,000)")

# image token cost alone (Gemini), exact for this sample
img_only=gem_ct("describe",image=True)-gem_ct("describe")
print(f"\n[Gemini image tokens for {IMG}: ~{img_only}]")

STORY=("Alex woke up early and ran to the park with a red ball. His friend Mia was already there, "
"swinging high. [emotion:QT/happy] Alex smiled and shouted hello. They played catch, but the ball "
"rolled under a big green bush. They looked everywhere. A friendly dog named Spot found the ball and "
"carried it back. [emotion:QT/surprised] Everyone laughed. They shared snacks. Alex learned that "
"asking a friend for help makes hard things easy and fun.")

# (provider, model, system_or_None, text, image?)
items=[
 ("AI Conversation Assistant",'claude','claude-sonnet-4-6',SAR, "What is your favourite color? (Alex (Age: 5))", False),
 ("ASR intent correction",'claude','claude-sonnet-4-6',
   "You correct ASR mishearings for a child's therapy robot. Decide if the transcript likely intended the target word(s) given the immediate context. Be conservative; only match when highly likely. Respond strictly in compact JSON.",
   "Expected: 'banana'\nHeard: 'banaana'\nContext: 'find the yellow fruit'\nAnswer in JSON only with keys match and canonical.", False),
 ("Story: comprehension Qs",'gemini','gemini-2.5-flash',None,
   "You generate comprehension questions for children's stories. Return JSON only.\nYou are creating comprehension questions for a story read by a 5-year-old named Alex.\n\nStory:\n"+STORY+"\n\nGenerate exactly 4 questions: 1 main idea + 3 detail. Simple language for 5-6. Short options. Each: 1 correct + 2 wrong. Return ONLY a JSON array.", False),
 ("Story: takeaway MCQs",'gemini','gemini-2.5-flash',None,
   "You write children's multiple-choice comprehension questions. Return JSON only.\nFor a 5-year-old named Alex; 3 takeaways; one question each.\n\nStory:\n"+STORY+"\n\nTakeaways: 1.\"Asking for help makes hard things easier.\" 2.\"Sharing is kind.\" 3.\"Working together is fun.\" Return a JSON array of 3 objects.", False),
 ("Story: gesture/emotion tagging",'gemini','gemini-2.5-flash',None,
   "You add inline gesture/emotion tags to children's stories. Return only the tagged story.\nReturn the SAME story word-for-word with [gesture:NAME]/[emotion:NAME] tags. Allowed emotions QT/happy,QT/sad,QT/surprised,QT/afraid,QT/angry,QT/calm,QT/shy. Allowed gestures hi,bye,nodding-yes,clapping,hoora,happy,calm,shy,embrace,patience,slight_no,think,sneezing,yawn,breathing_exercise,kiss,stretching. Tag every beat; do not change words.\n\nSTORY:\n"+STORY, False),
 ("Story: page splitting",'gemini','gemini-2.5-flash',None,
   "You split stories into pages. Return JSON only.\nSplit for a 5-year-old; about 2 to 3 sentences/page; keep sentences intact; preserve tags; return a JSON array of strings.\n\nStory:\n"+STORY, False),
 ("Story: scene identification",'gemini','gemini-2.5-flash',None,
   "You analyze story structure. Return JSON only.\n4 paragraphs; decide scene per paragraph; default each is its own scene; return scenes[] and chunk_to_scene[].\n\nP1: "+STORY[:110]+"\nP2: "+STORY[110:220]+"\nP3: "+STORY[220:330]+"\nP4: "+STORY[330:], False),
 ("Quiz: generation",'gemini','gemini-2.5-flash',None,
   "You create short, child-friendly quiz questions. Return JSON only.\nAct as a pediatric educator. Create 5 questions about 'animals'. Target ages 4-5. Types: yes_no, wh. Objective true/false; under 8 words. Return ONE JSON array with question/type/correct_answer/accepted_answers.", False),
 ("Quiz: feedback",'gemini','gemini-2.5-flash',None,
   "You create short, child-friendly quiz questions. Return JSON only.\nYou are a socially assistive robot in a pediatric therapeutic setting. Here is your system prompt for context:\n"+SAR+"\n\nGenerate 10 correct + 10 incorrect feedback phrases (2-8 words, warm, no emojis). Return JSON only.", False),
 ("Quiz: WH options",'gemini','gemini-2.5-flash',None,
   "You create short, child-friendly quiz questions. Return JSON only.\nFor each question generate exactly 3 plausible-but-WRONG options (1-3 words). Return JSON list of lists.\nInput: "+'[{"question":"Where do dogs sleep?","correct_answer":"bed"},{"question":"What does a cat say?","correct_answer":"meow"}]', False),
 ("Scene game: criteria (4-6)",'gemini','gemini-2.5-flash',None,
   "You generate game questions for children. Return JSON only.\nObject detection game for a 5-year-old. Toys: banana, red car, green dinosaur, tomato, blue block, ball, pencil, carrot.\n"+kb.build_question_prompt_fragment(5,'boy')+"\nGenerate ONE inference request; do NOT name the target/toys; criteria one noun + at most one adjective. Return JSON object question/criteria.", False),
 ("Scene game: riddle (7+)",'gemini','gemini-2.5-flash',None,
   "You generate game questions for children. Return JSON only.\nObject detection game for an 8-year-old. Toys: banana, red car, green dinosaur, tomato, blue block, ball, pencil, carrot.\n"+kb.build_question_prompt_fragment(8,'boy')+"\nGenerate ONE riddle; never name target/toys; pronouns+properties. Return JSON object question/criteria.", False),
 ("Scene game: spatial (still)",'gemini','gemini-2.5-flash',None,
   "You are judging a children's spatial-direction game.\nValid objects: banana, blue block.\nThe child was asked to put the banana under the blue block. Decide presence of each, the actual relation (next_to/above/under/behind/in_front_of/in/out/other), and whether it matches 'under'. Return ONLY a JSON object obj_a_found/obj_b_found/actual_relation/correct/reason.", True),
 ("Scene game: object detection",'gemini-robotics','gemini-robotics-er-1.6-preview',None,
   "Point to no more than 1 item a person is holding in the image. Return the object's identifying name, its dominant color, and its shape. The answer should follow the json format: [{\"point\": <point>, \"label\": <label>, \"color\": <color>, \"shape\": <shape>}]. The points are in [y, x] format normalized to 0-1000.", True),
 ("Toy recovery question",'gemini','gemini-2.5-flash',None,
   "Look at this image from a robot's camera. A child may be holding a toy. Identify it and say a short warm sentence mentioning it and asking a simple question. Age-appropriate for a 5-year-old (bands for 2-3, 4-5, 6-7, 8+). If no object, gentle prompt. Return JSON object text/object.\n\nThe child's name is Alex.", True),
 ("Conversation follow-up",'gemini','gemini-2.5-flash',None,
   "You are a friendly social robot having a conversation with a child.\nCONTEXT: Theme favourite animals; name Alex; age 5; follow-up 2 of 3.\nConversation: Robot: What animals do you like? Child: I like dogs. Robot: Do you have a dog? The child said: \"yes a brown one\". Acknowledge then respond.\nTASK: warm response. Age bands 2-3/4-5/6-7/8+. Stay on theme. End with a simple question. Return JSON object text/acknowledged.", False),
 ("WH Picture Scene: receptive",'gemini','gemini-2.5-flash',None,
   "You are a pediatric speech-language pathologist creating therapy materials. Focus only on the illustration on the card; ignore hands/person/table/background. Generate WH-questions for a child aged 5: for who/what/when/where/why give a question, correct answer (1-5 words), four visual choices (1-4 words), an evidence hint. Receptive: simple, obvious choices. Return ONLY valid JSON.", True),
 ("WH Picture Scene: expressive",'gemini','gemini-2.5-flash',None,
   "You are a pediatric speech-language pathologist creating therapy materials. Focus only on the illustration on the card. Generate 5 OPEN-ENDED imagination questions for a child aged 5 (FUTURE/PAST/PERSONAL/ALTERNATIVE/FEELING); no right/wrong; wh_type what/when/why; answer empty; visual_choices empty; evidence_hint a short imagination prompt. Return ONLY valid JSON.", True),
 ("Story sentence illustration",'gemini-img','gemini-2.5-flash-image',None,
   "You are an illustrator. Use a consistent, children's-book illustration style: soft round shapes, pastel color palette, thick outlines, minimal shading. The image should feel warm and friendly.use the given image as a reference to keep the style the exact same, it is not reevant otherwise the image generation.do not describe the image\n\nAlex and Mia played catch together in the sunny park with a red ball.", True),
]

print("\n=== EXACT prompt tokens (count_tokens) ===")
print(f"{'Activity':38} {'Model':32} {'tok':>6} {'+img':>5} {'%ctx':>7}")
for name,prov,model,system,text,has_img in items:
    try:
        if prov=='claude':
            tok=claude_ct(system,text); imgtok=0
        elif prov in ('gemini','gemini-robotics'):
            base=gem_ct(text,model=model)
            tok=base; imgtok=(gem_ct(text,model=model,image=True)-base) if has_img else 0
        else: # gemini-img: count on flash (image model may not support count_tokens), note
            try:
                base=gem_ct(text,model='gemini-2.5-flash-image')
            except Exception:
                base=gem_ct(text)  # fallback tokenizer proxy
            tok=base; imgtok=img_only if has_img else 0
        total=tok+imgtok; c=CTX[model]
        print(f"{name:38} {model:32} {tok:>6} {imgtok:>5} {100*total/c:>6.3f}%")
    except Exception as e:
        print(f"{name:38} {model:32}  ERR {type(e).__name__}: {str(e)[:60]}")
