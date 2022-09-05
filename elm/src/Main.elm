module Main exposing (..)

import Browser
import Classification exposing (Classification, classDecoder)
import File exposing (File)
import File.Select as Select
import Html exposing (Html, button, div, h1, img, text)
import Html.Attributes exposing (class, src)
import Html.Events exposing (onClick)
import Http
import RemoteData exposing (WebData)
import Request
import Task


type alias Model =
    { result : WebData Classification
    , imagePreview : String
    , image : Maybe File
    }


type Msg
    = GotResponse (WebData Classification)
    | SelectImage
    | ImageLoaded File
    | ImagePreviewLoaded String
    | UploadImage


upload : File -> Cmd Msg
upload image =
    Http.request
        { method = "POST"
        , url = "/predict"
        , headers = []
        , body =
            Http.multipartBody
                [ Http.filePart "file" image
                ]
        , expect = Request.expectJson (RemoteData.fromResult >> GotResponse) classDecoder
        , timeout = Nothing
        , tracker = Nothing
        }


init : () -> ( Model, Cmd Msg )
init _ =
    let
        model =
            { result = RemoteData.NotAsked
            , imagePreview = ""
            , image = Nothing
            }
    in
    ( model
    , Cmd.none
    )


results : Classification -> Html Msg
results result =
    let
        result_ =
            "Results: " ++ result.class
    in
    div [ class "results" ] [ text result_ ]


imagePreview : String -> Html Msg
imagePreview image_url =
    img [ class "sample", src image_url ] []


uploader : Html Msg
uploader =
    div [ class "uploader", onClick SelectImage ]
        [ img [ src "/static/images/upload-cloud.svg", class "upload" ] []
        ]


submitButton : String -> Html Msg
submitButton state =
    case state of
        "active" ->
            button [ class state, onClick UploadImage ] [ text "Check for Melanoma" ]

        _ ->
            button [ class state ] [ text "Check for Melanoma" ]


view : Model -> Html Msg
view model =
    div [ class "content" ]
        [ h1 [] [ text "Melanoma Detection System" ]
        , div [ class "main" ]
            [ case model.image of
                Just img ->
                    imagePreview model.imagePreview

                Nothing ->
                    uploader
            , case model.image of
                Just image ->
                    case model.result of
                        RemoteData.Success result ->
                            results result

                        _ ->
                            submitButton "active"

                Nothing ->
                    submitButton "disabled"
            ]
        ]


update : Msg -> Model -> ( Model, Cmd Msg )
update msg model =
    case msg of
        GotResponse result ->
            ( { model
                | result = result
              }
            , Cmd.none
            )

        SelectImage ->
            ( model, Select.file [ "image/jpg", "image/jpeg" ] ImageLoaded )

        ImageLoaded image ->
            let
                isFileTypeAlllowed =
                    [ "image/jpg", "image/jpeg" ]
                        |> List.member (File.mime image)
            in
            case isFileTypeAlllowed of
                True ->
                    ( { model
                        | image = Just image
                      }
                    , Task.perform ImagePreviewLoaded (File.toUrl image)
                    )

                False ->
                    ( model
                    , Cmd.none
                    )

        ImagePreviewLoaded image_url ->
            ( { model
                | imagePreview = image_url
              }
            , Cmd.none
            )

        UploadImage ->
            case model.image of
                Just image ->
                    ( model, upload image )

                Nothing ->
                    ( model, Cmd.none )


subscriptions : Model -> Sub Msg
subscriptions model =
    Sub.none


main =
    Browser.element
        { init = init
        , update = update
        , subscriptions = subscriptions
        , view = view
        }
