# Third-party notices

The root [MIT license](LICENSE) covers the original project code. The third-party materials below retain their own licenses, including when their text appears in CSVs, model prompts, qualitative examples or embedded notebook data.

## e-SNLI and SNLI text

The frozen examples come from [e-SNLI](https://github.com/OanaMariaCamburu/e-SNLI), revision `7b585a3f077fdea899780eb0473940522ae44a2e`. e-SNLI was introduced by Oana-Maria Camburu, Tim Rocktäschel, Thomas Lukasiewicz and Phil Blunsom in *e-SNLI: Natural Language Inference with Natural Language Explanations* (2018). Its repository supplies the MIT notice reproduced below.

The underlying sentence pairs originate in [SNLI](https://nlp.stanford.edu/projects/snli/), by Samuel R. Bowman, Gabor Angeli, Christopher Potts and Christopher D. Manning (2015). Stanford distributes the corpus under [Creative Commons Attribution-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-sa/4.0/). SNLI includes Flickr30k captions, attributed to Peter Young, Alice Lai, Micah Hodosh and Julia Hockenmaier (2014), as described on the SNLI project page.

The SNLI-derived text and its counterfactual adaptations in this repository are distributed under CC BY-SA 4.0. Project modifications comprise selecting 1,000 examples, inserting or deleting recorded target spans, and adding experimental annotations. Source IDs, original text and edit spans are retained in `data/`; the code license does not replace the source-text terms.

### e-SNLI notice

Source: [LICENSE at the pinned e-SNLI revision](https://github.com/OanaMariaCamburu/e-SNLI/blob/7b585a3f077fdea899780eb0473940522ae44a2e/LICENSE).

```text
Copyright <2019> <Oana-Maria Camburu>

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
```

## VADER lexicon

`cache/nltk/sentiment/vader_lexicon.zip` is the VADER sentiment lexicon distributed through NLTK. The original lexicon is by C.J. Hutto and is used for target-polarity annotations. The bundled archive is unmodified. Upstream: [VADER](https://github.com/cjhutto/vaderSentiment), [license](https://github.com/cjhutto/vaderSentiment/blob/master/LICENSE.txt).

```text
The MIT License (MIT)

Copyright (c) 2016 C.J. Hutto

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
