from typing import List
from pytube import cipher
from pytube.cipher import Cipher, logger, re, RegexMatchError, get_transform_plan, get_transform_map, get_throttling_plan, get_throttling_function_array
from pytube import YouTube, Caption, exceptions
from pytube.innertube import InnerTube, _default_clients

# def get_throttling_function_name(js: str) -> str:
#     """Extract the name of the function that computes the throttling parameter.

#     :param str js:
#         The contents of the base.js asset file.
#     :rtype: str
#     :returns:
#         The name of the function used to compute the throttling parameter.
#     """
#     function_patterns = [
#         # https://github.com/ytdl-org/youtube-dl/issues/29326#issuecomment-865985377
#         # https://github.com/yt-dlp/yt-dlp/commit/48416bc4a8f1d5ff07d5977659cb8ece7640dcd8
#         # var Bpa = [iha];
#         # ...
#         # a.C && (b = a.get("n")) && (b = Bpa[0](b), a.set("n", b),
#         # Bpa.length || iha("")) }};
#         # In the above case, `iha` is the relevant function name
#         #r'a\.[a-zA-Z]\s*&&\s*\([a-z]\s*=\s*a\.get\("n"\)\)\s*&&\s*'
#         r'a\.[a-zA-Z]\s*&&\s*\([a-z]\s*=\s*a\.get\("n"\)\)\s*&&.*?\|\|\s*([a-z]+)',
#         r'\([a-z]\s*=\s*([a-zA-Z0-9$]+)(\[\d+\])?\([a-z]\)',
#         # r'a.[a-zA-Z]\s*&&\s*([a-z]\s*=\sa.get("n"))\s&&\s*',
#         # r'([a-z]\s*=\s*([a-zA-Z0-9$]{2,3})([\d+])?([a-z])'
#     ]
#     logger.debug('Finding throttling function name')
#     for pattern in function_patterns:
#         regex = re.compile(pattern)
#         function_match = regex.search(js)
#         if function_match:
#             logger.debug("finished regex search, matched: %s", pattern)
#             if len(function_match.groups()) == 1:
#                 return function_match.group(1)
#             idx = function_match.group(2)
#             if idx:
#                 idx = idx.strip("[]")
#                 array = re.search(
#                     r'var {nfunc}\s*=\s*(\[.+?\]);'.format(
#                         nfunc=re.escape(function_match.group(1))),
#                     js
#                 )
#                 if array:
#                     array = array.group(1).strip("[]").split(",")
#                     array = [x.strip() for x in array]
#                     return array[int(idx)]

#     raise RegexMatchError(
#         caller="get_throttling_function_name", pattern="multiple"
#     )

# cipher.get_throttling_function_name = get_throttling_function_name





# def bypass_age_gate(self):
#     """Attempt to update the vid_info by bypassing the age gate."""
#     innertube = InnerTube(
#         client='ANDROID_EMBED',
#         use_oauth=self.use_oauth,
#         allow_cache=self.allow_oauth_cache
#     )
#     innertube_response = innertube.player(self.video_id)

#     playability_status = innertube_response['playabilityStatus'].get('status', None)

#     # If we still can't access the video, raise an exception
#     # (tier 3 age restriction)
#     if playability_status == 'UNPLAYABLE':
#         raise exceptions.AgeRestrictedError(self.video_id)

#     self._vid_info = innertube_response


# DEFAULT_CLIENTS = [
#     'ANDROID_EMBED',
#     'WEB',
#     'ANDROID',
#     'IOS',
#     'WEB_EMBED',    
#     'IOS_EMBED',
#     'WEB_MUSIC',
#     'ANDROID_MUSIC',
#     'IOS_MUSIC',
#     'WEB_CREATOR',
#     'ANDROID_CREATOR',
#     'IOS_CREATOR',
#     'MWEB',
#     'TV_EMBED'
# ]

DEFAULT_CLIENTS = [
    'ANDROID_EMBED',
    'ANDROID',
    'WEB_EMBED',
    'WEB',    
    'IOS_EMBED',    
    'IOS',            
    'WEB_MUSIC',
    'ANDROID_MUSIC',
    'IOS_MUSIC',
    'WEB_CREATOR',
    'ANDROID_CREATOR',
    'IOS_CREATOR',
    'MWEB',
    'TV_EMBED'
]


def bypass_age_gate(self):
    """Attempt to update the vid_info by bypassing the age gate."""
    for client_key in DEFAULT_CLIENTS:
        print(">>>>Trying client: " + client_key)
        innertube = InnerTube(
            # client='ANDROID_EMBED',
            client=client_key,
            use_oauth=self.use_oauth,
            allow_cache=self.allow_oauth_cache
        )
        innertube_response = innertube.player(self.video_id)

        playability_status = innertube_response['playabilityStatus'].get('status', None)

        # If we still can't access the video, raise an exception
        # (tier 3 age restriction)
        if playability_status == 'UNPLAYABLE':
            continue
            # raise exceptions.AgeRestrictedError(self.video_id)

        self._vid_info = innertube_response
        return 
    else:
        raise exceptions.AgeRestrictedError(self.video_id)


YouTube.bypass_age_gate = bypass_age_gate





def get_transform_object(js: str, var: str) -> List[str]:
    pattern = r"var %s={(.*?)};" % re.escape(var)
    logger.debug("getting transform object")
    regex = re.compile(pattern, flags=re.DOTALL)
    transform_match = regex.search(js)
    
    if not transform_match:
        # i commented out the line raising the error
        # raise RegexMatchError(caller="get_transform_object", pattern=pattern)
        logger.error(f"No match found for pattern: {pattern}")
        return []  # Return an empty list if no match is found

    return transform_match.group(1).replace("\n", " ").split(", ")



cipher.get_transform_object = get_transform_object




def cypher_init(self, js: str):
    self.transform_plan: List[str] = get_transform_plan(js)
    var_regex = re.compile(r"^\$\w+\W")
    var_match = var_regex.search(self.transform_plan[0])
    if not var_match:
        raise RegexMatchError(
            caller="__init__", pattern=var_regex.pattern
        )
    var = var_match.group(0)[:-1]
    self.transform_map = get_transform_map(js, var)
    self.js_func_patterns = [
        r"\w+\.(\w+)\(\w,(\d+)\)",
        r"\w+\[(\"\w+\")\]\(\w,(\d+)\)"
    ]

    self.throttling_plan = get_throttling_plan(js)
    self.throttling_array = get_throttling_function_array(js)

    self.calculated_n = None

# Cipher.__init__ = cypher_init