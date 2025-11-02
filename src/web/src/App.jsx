import React, { useState, useEffect } from "react";
import io from "socket.io-client";
import "./App.css";
import PrismaticBurst from "./js/prismatic-burst.jsx";
import Orb from "./js/orb.jsx";
import DarkVeil from "./js/DarkVeil.jsx";
import SplitText from "./js/SplitText.jsx";

const socket = io("http://localhost:5000");

function App() {
  const [veilVisible, setVeilVisible] = useState(true);
  const [orbExpanded, setOrbExpanded] = useState(false);
  const [orbVisible, setOrbVisible] = useState(true);
  const [orbOpacity, setOrbOpacity] = useState(1); // 👈 new for fade control
  const [showText, setShowText] = useState(false);
  const [isAnimating, setIsAnimating] = useState(false);

  const [songText, setSongText] = useState("");
  const [subtitleText, setSubtitleText] = useState("");

  const startReveal = () => {
    if (isAnimating) return;
    setIsAnimating(true);

    // Fade veil out
    setVeilVisible(false);

    // After 5 seconds, fade veil back in and expand orb
    setTimeout(() => {
      setVeilVisible(true);
      setOrbExpanded(true);

      // After orb expands, hide orb and show text
      setTimeout(() => {
        // Fade orb out
        setOrbOpacity(0);
        setTimeout(() => setOrbVisible(false), 1000);

        // Show text
        setShowText(true);

        // Keep text visible for 10 seconds, then reset
        setTimeout(() => {
          setShowText(false);
          setOrbExpanded(false);

          // Wait for fade transition, then show orb again
          setTimeout(() => {
            setOrbVisible(true);
            // 👇 make orb fade back in smoothly
            setTimeout(() => setOrbOpacity(1), 100); 
            setIsAnimating(false); // ready for next trigger
          }, 1500);
        }, 10000);
      }, 3000);
    }, 5000);
  };

  useEffect(() => {
    socket.on("trigger", (data) => {
      console.log("Received from Python:", data);

      if (isAnimating) {
        console.log("Ignoring trigger — still animating.");
        return;
      }

      // Update text from Python
      const song = data.song || data.title || "Unknown Song";
      const artist = data.artist || "Unknown Artist";
      const subtitle = data.subtitle || "Triggered";

      setSongText(`${song} - ${artist}`);
      setSubtitleText(subtitle);

      // Start the reveal animation
      startReveal();
    });

    return () => socket.off("trigger");
  }, [isAnimating]);

  return (
    <div
      style={{
        position: "absolute",
        inset: 0,
        overflow: "hidden",
        backgroundColor: "black",
      }}
    >
      {/* Background */}
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

      {/* Veil */}
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
            transition:
              "transform 3s ease-in-out, opacity 2s ease-in-out",
            opacity: orbOpacity, // 👈 smooth fade control
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

      {/* Text */}
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
          <SplitText
            text={songText}
            delay={70}
            duration={2}
            ease="elastic.out(1, 0.3)"
            splitType="chars"
            from={{ opacity: 0, y: 40 }}
            to={{ opacity: 0.6, y: 0 }}
            threshold={0.1}
            rootMargin="-100px"
            textAlign="center"
            className="split-text"
            style={{
              fontSize: "4rem",
              fontWeight: "bold",
              color: "white",
              fontFamily: "Helvetica, Arial, sans-serif",
            }}
          />

          <div
            style={{
              marginTop: "0.2rem",
              fontSize: "2rem",
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
    </div>
  );
}

export default App;

