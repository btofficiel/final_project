module Classification exposing (Classification, classDecoder)

import Json.Decode as Decode exposing (Decoder, int, list, string)
import Json.Decode.Pipeline exposing (optional, required)


type alias Classification =
    { class : String
    }


classDecoder : Decoder Classification
classDecoder =
    Decode.succeed Classification
        |> required "class" string
