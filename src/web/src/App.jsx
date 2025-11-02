import React, { useState, useEffect } from "react";
import io from "socket.io-client";
import "./App.css";
import PrismaticBurst from "./js/prismatic-burst.jsx";
import Orb from "./js/orb.jsx";
import DarkVeil from "./js/DarkVeil.jsx";
import SplitText from "./js/SplitText.jsx";
import GlassSurface from "./js/GlassSurface.jsx";

const socket = io("http://localhost:5000");

function App() {
  const [veilVisible, setVeilVisible] = useState(true);
  const [orbExpanded, setOrbExpanded] = useState(false);
  const [orbVisible, setOrbVisible] = useState(true);
  const [orbOpacity, setOrbOpacity] = useState(1);
  const [showText, setShowText] = useState(false);
  const [isAnimating, setIsAnimating] = useState(false);
  const [showResetButton, setShowResetButton] = useState(false);

  const [songTitle, setSongTitle] = useState("");
  const [artistName, setArtistName] = useState("");
  const [subtitleText, setSubtitleText] = useState("");

  /** Triggered by server event to reveal song info */
  const startReveal = () => {
    if (isAnimating) return;
    setIsAnimating(true);
    setShowResetButton(false);

    setVeilVisible(false);

    setTimeout(() => {
      setVeilVisible(true);
      setOrbExpanded(true);

      setTimeout(() => {
        setOrbOpacity(0);
        setTimeout(() => setOrbVisible(false), 1000);
        setShowText(true);
        setShowResetButton(true);
        setIsAnimating(false);
      }, 3000);
    }, 5000);
  };

  /** Reset everything back to idle smoothly */
  const resetToIdle = () => {
    if (isAnimating) return;
    setIsAnimating(true);

    setShowText(false);
    setShowResetButton(false);
    setOrbVisible(true);
    setOrbExpanded(false);
    setOrbOpacity(0);

    setTimeout(() => setOrbOpacity(1), 50);
    setVeilVisible(true);

    setTimeout(() => setIsAnimating(false), 1500);
    setTimeout(() => socket.emit("client_idle"), 1500);
  };

  /** Socket listener for song trigger */
  useEffect(() => {
    const handleTrigger = (data) => {
      if (isAnimating || showText) {
        console.log("Ignoring trigger — still animating or already showing.");
        return;
      }

      const song = data.song || data.title || "Unknown Song";
      const artist = data.artist || "Unknown Artist";
      const subtitle = data.subtitle || "Triggered";

      setSongTitle(song);
      setArtistName(artist);
      setSubtitleText(subtitle);

      startReveal();
    };

    socket.on("trigger", handleTrigger);
    return () => {
      socket.off("trigger", handleTrigger);
    };
  }, [isAnimating, showText]);

  return (
    <div
      style={{
        position: "absolute",
        inset: 0,
        overflow: "hidden",
        backgroundColor: "black",
      }}
    >
      {/* Background Prismatic Burst */}
      <PrismaticBurst
        animationType="rotate3d"
        intensity={1.5}
        speed={0.5}
        distort={3.6}
        paused={false}
        offset={{ x: 0, y: 0 }}
        hoverDampness={0.25}
        rayCount={30}
        mixBlendMode="lighten"
        colors={["#f0c8ff", "#773bcd", "#773bcd"]}
        style={{
          position: "absolute",
          inset: 0,
          opacity: veilVisible ? 0 : 1,
          transition: "opacity 1.5s ease-in-out",
        }}
      />

      {/* Dark Veil */}
      <div
        style={{
          position: "absolute",
          inset: 0,
          opacity: veilVisible ? 1 : 0,
          transition: "opacity 1.5s ease-in-out",
          pointerEvents: "none",
          zIndex: 10,
        }}
      >
        <DarkVeil />
      </div>

      {/* Orb */}
      {orbVisible && (
        <div
          style={{
            position: "absolute",
            top: "50%",
            left: "50%",
            transform: `translate(-50%, -50%) scale(${orbExpanded ? 5 : 1})`,
            transition: "transform 3s ease-in-out, opacity 2s ease-in-out",
            opacity: orbOpacity,
            width: "600px",
            height: "600px",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            zIndex: 20,
          }}
        >
          <Orb hoverIntensity={0.5} rotateOnHover={true} hue={0} />
        </div>
      )}

      {/* Song, Artist & Subtitle */}
      {showText && (
        <div
          style={{
            position: "absolute",
            top: "50%",
            left: "50%",
            transform: "translate(-50%, -50%)",
            textAlign: "center",
            zIndex: 25,
            transition: "opacity 1s ease-in-out",
          }}
        >
          {/* Combined song + artist on one line */}
          <div
            style={{
              fontSize: "4rem",
              fontFamily: "Helvetica, Arial, sans-serif",
              color: "white",
              opacity: 0,
              animation: "fadeIn 2s ease forwards",
              animationDelay: "1s",
            }}
          >
            <span
              style={{
                fontWeight: "400", // song name less bold
                opacity: 0.85,
              }}
            >
              {songTitle}
            </span>
            <span
              style={{
                fontWeight: "800", // artist more bold
                marginLeft: "0.4rem",
                textShadow:
                  "0 0 25px rgba(138,43,226,0.8), 0 0 45px rgba(66,11,186,0.6)",
              }}
            >
              {"– " + artistName}
            </span>
          </div>

          {/* Subtitle (confidence) */}
          <div
            style={{
              marginTop: "0.4rem",
              fontSize: "1.8rem",
              fontWeight: "300",
              fontFamily: "Helvetica, Arial, sans-serif",
              color: "white",
              opacity: 0,
              animation: "fadeIn 2s ease forwards",
              animationDelay: "2.5s",
            }}
          >
            {subtitleText}
          </div>
        </div>
      )}

      {/* Reset Button */}
      {showResetButton && (
        <div
          style={{
            position: "absolute",
            bottom: "3rem",
            left: "50%",
            transform: "translateX(-50%)",
            zIndex: 30,
            opacity: 0,
            animation: "fadeIn 2s ease forwards",
            animationDelay: "2.5s",
            pointerEvents: "auto",
          }}
        >
          <div
            onClick={resetToIdle}
            style={{
              display: "flex",
              justifyContent: "center",
              alignItems: "center",
              cursor: "pointer",
              width: "250px",
              height: "60px",
              borderRadius: "30px",
              background:
                "linear-gradient(135deg, rgba(66,11,186,0.7), rgba(138,43,226,0.4))",
              boxShadow:
                "0 0 25px rgba(66,11,186,0.8), 0 0 50px rgba(138,43,226,0.6)",
              backdropFilter: "blur(25px)",
              border: "1px solid rgba(255,255,255,0.15)",
              position: "relative",
              overflow: "hidden",
              pointerEvents: "auto",
            }}
          >
            <span
              style={{
                fontSize: "1.5rem",
                fontWeight: "bold",
                color: "white",
                fontFamily: "Helvetica, Arial, sans-serif",
                textShadow:
                  "0 0 12px rgba(66,11,186,0.9), 0 0 18px rgba(138,43,226,0.7)",
              }}
            >
              Go Back
            </span>

            <div
              style={{
                content: '""',
                position: "absolute",
                top: 0,
                left: "-50%",
                width: "50%",
                height: "100%",
                background:
                  "linear-gradient(120deg, rgba(255,255,255,0.3), rgba(255,255,255,0))",
                transform: "skewX(-20deg)",
                animation: "shine 5s infinite",
              }}
            />
          </div>
        </div>
      )}
    </div>
  );
}

export default App;

